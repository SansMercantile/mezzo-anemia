# sans-mercantile-app/backend/mezzo_anima_line/digital_legacy_manager.py
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import hashlib
import json
import io # Added for PyPDF2
import base64 # Existing for Fernet
from cryptography.fernet import Fernet # Existing for encryption

from firebase_admin import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter
from backend.config.settings import settings
from backend.mezzo_anima_line.mezzo_agent_protocol import MezzoMemoryRecord
from mpeti_modules.ai_ops import LLMClient
from backend.dependencies import get_singleton # Used for ZKVerifier
from backend.governance.tokenization.zk_verifier import ZKVerifier # Existing for ZKP

# Added PyPDF2 for PDF parsing
import PyPDF2 # You need to add PyPDF2 to your requirements.txt for Mezzo app

logger = logging.getLogger(__name__)

class DigitalLegacyManager:
    """
    Manages the secure ingestion, encryption, and conditional release of digital assets
    and legal documents, forming the core of Mezzo's digital legacy functionality.
    This implementation uses cryptographic encryption and Zero-Knowledge Proofs for verification.
    """
    def __init__(self, firestore_db: firestore.Client, message_broker):
        self.firestore_db = firestore_db
        self.message_broker = message_broker
        self.legacy_collection_ref = self.firestore_db.collection("digital_legacy_vault")
        self.storage_bucket_name = settings.DIGITAL_LEGACY_BUCKET # e.g., "mezzo-digital-legacy-vault"
        self.llm_client = LLMClient()
        self.zk_verifier: ZKVerifier = get_singleton("zk_verifier") # Get existing ZKVerifier singleton
        self.encryption_key = self._load_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        logger.info("DigitalLegacyManager initialized.")

    def _load_encryption_key(self) -> bytes:
        """
        Loads the encryption key from a secure source. In a production environment,
        this key would be managed and rotated by a Key Management Service (KMS).
        For this implementation, we derive a stable key from a seed in the settings.
        """
        if not settings.MEZZO_ENCRYPTION_KEY_SEED:
            raise ValueError("MEZZO_ENCRYPTION_KEY_SEED must be set in the environment configuration.")
        
        key_seed = settings.MEZZO_ENCRYPTION_KEY_SEED.encode()
        hashed_seed = hashlib.sha256(key_seed).digest()
        return base64.urlsafe_b64encode(hashed_seed)

    async def ingest_legal_document(self, user_id: str, document_content: bytes, metadata: Dict[str, Any], release_conditions: Optional[Dict[str, Any]] = None) -> str:
        """
        Securely ingests, encrypts, and stores a legal document (e.g., a will) with
        verifiable release conditions. The raw document content is encrypted and stored in Cloud Storage.
        Metadata, parsed summary, and encrypted reference are stored in Firestore.
        """
        doc_hash = hashlib.sha256(document_content).hexdigest()
        document_id = f"legal_doc_{user_id}_{datetime.utcnow().timestamp()}_{doc_hash[:8]}"
        
        # 1. Encrypt the raw document content
        encrypted_document_content = self.cipher_suite.encrypt(document_content)

        # 2. Upload encrypted content to Cloud Storage
        bucket = storage.bucket(self.storage_bucket_name)
        blob = bucket.blob(f"{user_id}/{document_id}/{metadata.get('filename', 'document.pdf')}.encrypted")
        
        try:
            await asyncio.to_thread(blob.upload_from_string, encrypted_document_content, content_type='application/octet-stream')
            logger.info(f"Encrypted document {document_id} uploaded to Cloud Storage path: {blob.name}")

            # 3. Parse document content (from original content before encryption for LLM processing)
            parsed_content = await self._parse_legal_document_content(document_content, metadata.get('content_type', 'application/pdf'))

            # 4. Store metadata and encrypted reference in Firestore
            doc_ref = self.legacy_collection_ref.document(document_id)
            doc_data = {
                "user_id": user_id,
                "document_id": document_id,
                "filename": metadata.get('filename'),
                "content_type": metadata.get('content_type'),
                "storage_path": blob.name, # Path to the encrypted blob in GCS
                "upload_timestamp": datetime.utcnow(),
                "status": "ingested",
                "access_conditions": release_conditions or {}, # Use provided release_conditions
                "release_keys": {}, # These would be references to keys, not actual keys
                "metadata": metadata,
                "parsed_summary": parsed_content.get("summary"),
                "extracted_entities": parsed_content.get("extracted_entities"),
                "key_clauses": parsed_content.get("key_clauses")
            }
            await asyncio.to_thread(doc_ref.set, doc_data)
            logger.info(f"Metadata for document {document_id} stored in Firestore.")

            await self.store_legacy_event_as_memory(user_id, document_id, "legal_document_ingested", doc_data)

            return document_id
        except Exception as e:
            logger.error(f"Failed to ingest legal document for user {user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to upload document: {e}")

    async def store_memory(self, user_id: str, memory_data: Dict[str, Any], release_conditions: Dict[str, Any]) -> str:
        """
        Encrypts and stores a digital memory with specified ZKP-based release conditions.
        """
        memory_json = json.dumps(memory_data).encode('utf-8')
        encrypted_memory = self.cipher_suite.encrypt(memory_json)
        
        mem_ref = self.firestore_db.collection("users", user_id, "digital_legacy").document()
        await asyncio.to_thread(mem_ref.set, {
            "type": "memory",
            "encrypted_data_b64": base64.b64encode(encrypted_memory).decode('utf-8'),
            "release_conditions": release_conditions,
            "stored_at": firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Successfully stored and encrypted memory {mem_ref.id} for user {user_id}.")
        return mem_ref.id

    async def _parse_legal_document_content(self, document_content: bytes, content_type: str) -> Dict[str, Any]:
        """
        Parses the content of a legal document, extracting text for LLM processing.
        Handles common document types like PDF, text, and JSON.
        """
        logger.info(f"Parsing legal document content (type: {content_type})...")
        
        text_content: str = ""
        
        try:
            if "pdf" in content_type:
                pdf_file = io.BytesIO(document_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() or ""
                logger.info("PDF content extracted using PyPDF2.")
            elif "text" in content_type:
                text_content = document_content.decode('utf-8', errors='ignore')
            elif "json" in content_type:
                text_content = json.dumps(json.loads(document_content.decode('utf-8')), indent=2)
            else:
                logger.warning(f"Unsupported document content type for direct text extraction: {content_type}. Attempting raw text interpretation.")
                text_content = document_content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error extracting text from document content type {content_type}: {e}", exc_info=True)
            text_content = f"Error extracting text: {e}. Raw content snippet: {document_content[:200].decode('utf-8', errors='ignore')}..."

        llm_input_text = text_content[:settings.LLM_MAX_PARSE_CHARS] if hasattr(settings, 'LLM_MAX_PARSE_CHARS') else text_content[:10000]

        prompt = (
            f"Analyze the following legal document content. Extract key entities (people, organizations, dates, asset names, clauses, beneficiaries, executors) "
            f"and provide a concise summary of its purpose and main provisions. "
            f"Format the output as a JSON object with 'summary' (string), 'extracted_entities' (list of strings), and 'key_clauses' (list of strings).\n\n"
            f"Document Content:\n{llm_input_text}"
        )

        try:
            llm_response_str = await self.llm_client.generate_content(prompt)
            parsed_data = json.loads(llm_response_str)
            
            if not isinstance(parsed_data, dict) or "summary" not in parsed_data or "extracted_entities" not in parsed_data or "key_clauses" not in parsed_data:
                raise ValueError("LLM response not in expected JSON format or missing required fields.")

            logger.info("Legal document parsed using LLM.")
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"LLM response for document parsing was not valid JSON: {llm_response_str[:200]}... Error: {e}")
            return {
                "document_type": content_type,
                "extracted_entities": [],
                "key_clauses": [],
                "summary": f"Failed to parse document content due to LLM response format. Raw LLM response: {llm_response_str[:500]}"
            }
        except Exception as e:
            logger.error(f"Error parsing legal document content with LLM: {e}", exc_info=True)
            return {
                "document_type": content_type,
                "extracted_entities": [],
                "key_clauses": [],
                "summary": f"Failed to parse document content: {e}"
            }

    async def ingest_life_history_data(self, user_id: str, data_type: str, content: Dict[str, Any], tags: List[str] = None) -> str:
        """
        Ingests various forms of life history data (interviews, voice notes, videos).
        """
        record_id = f"life_hist_{user_id}_{datetime.utcnow().timestamp()}"
        record = MezzoMemoryRecord(
            record_id=record_id,
            user_id=user_id,
            agent_id="MezzoMaterna", # Or specific agent handling ingestion
            type=data_type,
            content=content,
            tags=tags or ["life_history"]
        )
        try:
            doc_ref = self.firestore_db.collection(f"users/{user_id}/mezzo_memories").document(record.record_id)
            await asyncio.to_thread(doc_ref.set, record.to_dict())
            logger.info(f"Life history data '{data_type}' stored for user {user_id}.")
            return record_id
        except Exception as e:
            logger.error(f"Failed to store life history data for user {user_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to ingest life history data: {e}")

    async def verify_and_release_assets(self, user_id: str, claimant_id: str, proof_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verifies a claimant's request using Zero-Knowledge Proofs and releases the
        corresponding digital assets if the proof is valid. This is a real,
        cryptographically secure verification process.
        """
        asset_id = proof_payload.get("asset_id")
        if not asset_id:
            logger.warning("Claim rejected: No asset_id provided in proof payload.")
            return []

        asset_ref = self.firestore_db.collection("digital_legacy_vault").document(asset_id) # Corrected collection reference
        asset_doc = await asyncio.to_thread(asset_ref.get)

        if not asset_doc.exists:
            logger.warning(f"Claim rejected: Asset {asset_id} not found for user {user_id}.")
            return []

        asset_data = asset_doc.to_dict()
        release_conditions = asset_data.get("access_conditions", {})

        # 1. Verify Claimant Identity (from release_conditions)
        if release_conditions.get("claimant_id") and release_conditions.get("claimant_id") != claimant_id:
            logger.warning(f"Claim rejected: Claimant {claimant_id} is not authorized to access asset {asset_id} based on claimant_id condition.")
            return []

        # 2. Verify Zero-Knowledge Proof (if required by release_conditions)
        is_proof_valid = False
        if release_conditions.get("zk_circuit_id"):
            is_proof_valid = await self.zk_verifier.verify_proof(
                circuit_id=release_conditions.get("zk_circuit_id"),
                proof=proof_payload.get("proof"),
                public_inputs=proof_payload.get("public_inputs")
            )
            if not is_proof_valid:
                logger.warning(f"Claim rejected: Zero-Knowledge Proof verification failed for asset {asset_id}.")
                return []
            logger.info(f"ZKP verification successful for asset {asset_id}.")
        else:
            # If no ZKP required, assume this step passes for now, or add other non-ZKP conditions
            is_proof_valid = True # No ZKP required, so consider this part valid

        # 3. Verify other Access Conditions (e.g., age, date)
        if not self._verify_access_conditions(release_conditions, verification_data):
            logger.warning(f"Access conditions not met for document {document_id} for user {user_id}.")
            return []

        # 4. Verify Key Release Mechanism (biometric, PIN, or notarized approval)
        if not await self._verify_key_release(doc_data.get("release_keys", {}), verification_data): # Use doc_data from Firestore
            logger.warning(f"Key release verification failed for document {document_id} for user {user_id}.")
            return []

        # 5. Decrypt and Release Asset
        bucket = storage.bucket(self.storage_bucket_name)
        blob = bucket.blob(asset_data["storage_path"]) # Path to the encrypted blob in GCS
        
        try:
            encrypted_content = await asyncio.to_thread(blob.download_as_bytes)
            logger.info(f"Encrypted content for asset {asset_id} downloaded from Cloud Storage.")
            
            decrypted_data_bytes = self.cipher_suite.decrypt(encrypted_content)
            logger.info(f"Asset {asset_id} decrypted for user {user_id}.")
            
            # Update status in Firestore
            await asyncio.to_thread(asset_ref.update, {"status": "released", "last_released": datetime.utcnow()})
            await self.store_legacy_event_as_memory(user_id, asset_id, "asset_released_via_zkp", {"document_id": asset_id, "access_type": verification_data.get("type"), "claimant_id": claimant_id})

            # Return the decrypted content and metadata
            return [{
                "metadata": asset_data.get("metadata"),
                "decrypted_data_b64": base64.b64encode(decrypted_data_bytes).decode('utf-8')
            }]

        except Exception as e:
            logger.error(f"Failed to decrypt/release asset {asset_id} for user {user_id} even after successful verification. Error: {e}", exc_info=True)
            return []

    async def store_memory(self, user_id: str, memory_data: Dict[str, Any], release_conditions: Dict[str, Any]) -> str:
        """
        Encrypts and stores a digital memory with specified ZKP-based release conditions.
        """
        memory_json = json.dumps(memory_data).encode('utf-8')
        encrypted_memory = self.cipher_suite.encrypt(memory_json)
        
        mem_ref = self.firestore_db.collection("users", user_id, "digital_legacy").document()
        await asyncio.to_thread(mem_ref.set, {
            "type": "memory",
            "encrypted_data_b64": base64.b64encode(encrypted_memory).decode('utf-8'),
            "release_conditions": release_conditions,
            "stored_at": firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Successfully stored and encrypted memory {mem_ref.id} for user {user_id}.")
        return mem_ref.id

    async def release_digital_asset(self, user_id: str, document_id: str, verification_data: Dict[str, Any]) -> bytes: # Changed return type to bytes
        """
        Handles the conditional release of a digital asset based on predefined conditions
        and biometric/PIN/notarized approval.
        Returns the decrypted raw byte content of the document.
        """
        doc_ref = self.legacy_collection_ref.document(document_id)
        doc_snapshot = await asyncio.to_thread(doc_ref.get)

        if not doc_snapshot.exists:
            raise HTTPException(status_code=404, detail="Digital asset not found.")

        doc_data = doc_snapshot.to_dict()
        if doc_data["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized access to digital asset.")

        access_conditions = doc_data.get("access_conditions", {})
        
        if not self._verify_access_conditions(access_conditions, verification_data):
            logger.warning(f"Access conditions not met for document {document_id} for user {user_id}.")
            raise HTTPException(status_code=403, detail="Access conditions not met.")

        if not await self._verify_key_release(doc_data.get("release_keys", {}), verification_data):
            logger.warning(f"Key release verification failed for document {document_id} for user {user_id}.")
            raise HTTPException(status_code=403, detail="Key release verification failed.")

        bucket = storage.bucket(self.storage_bucket_name)
        blob = bucket.blob(doc_data["storage_path"])
        
        try:
            encrypted_content = await asyncio.to_thread(blob.download_as_bytes)
            logger.info(f"Encrypted content for document {document_id} downloaded from Cloud Storage.")
            
            decrypted_data_bytes = self.cipher_suite.decrypt(encrypted_content)
            logger.info(f"Document {document_id} decrypted for user {user_id}.")

            await asyncio.to_thread(doc_ref.update, {"status": "released", "last_released": datetime.utcnow()})
            await self.store_legacy_event_as_memory(user_id, document_id, "asset_released", {"document_id": document_id, "access_type": verification_data.get("type")})

            return decrypted_data_bytes
        except Exception as e:
            logger.error(f"Failed to generate signed URL for document {document_id}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to release digital asset: {e}")

    def _verify_access_conditions(self, conditions: Dict[str, Any], verification_data: Dict[str, Any]) -> bool:
        """
        Verifies if the access conditions for a document are met.
        """
        if conditions.get("type") == "age":
            required_age = conditions.get("value")
            user_age = verification_data.get("user_age")
            if user_age is None or user_age < required_age:
                logger.warning(f"Age condition not met. Required: {required_age}, User: {user_age}")
                return False
        return True

    async def _verify_key_release(self, release_keys: Dict[str, Any], verification_data: Dict[str, Any]) -> bool:
        """
        Verifies the key release mechanism (biometric, PIN, or notarized approval).
        """
        verification_type = verification_data.get("type")
        verification_value = verification_data.get("value")

        if verification_type == "biometric":
            stored_biometric_hash = release_keys.get("biometric_hash")
            if stored_biometric_hash and stored_biometric_hash == verification_value:
                logger.info("Biometric verification successful.")
                return True
            logger.warning("Biometric verification failed.")
            return False
        elif verification_type == "pin":
            stored_pin_hash = release_keys.get("pin_hash")
            if stored_pin_hash and hashlib.sha256(verification_value.encode()).hexdigest() == stored_pin_hash:
                logger.info("PIN verification successful.")
                return True
            logger.warning("PIN verification failed.")
            return False
        elif verification_type == "notarized_approval":
            approval_record_id = verification_data.get("approval_record_id")
            if approval_record_id:
                approval_doc = await asyncio.to_thread(self.firestore_db.collection("notarized_approvals").document(approval_record_id).get)
                if approval_doc.exists and approval_doc.to_dict().get("status") == "approved":
                    logger.info("Notarized approval successful.")
                    return True
            logger.warning("Notarized approval verification failed.")
            return False
        
        logger.warning(f"Unknown verification type: {verification_type}")
        return False

    async def store_legacy_event_as_memory(self, user_id: str, event_id: str, event_type: str, content: Dict[str, Any], tags: List[str] = None):
        """
        Stores a digital legacy event as a MezzoMemoryRecord for the user's long-term memory.
        """
        record = MezzoMemoryRecord(
            user_id=user_id,
            agent_id=self.__class__.__name__,
            type=event_type,
            content=content,
            tags=tags or ["digital_legacy"],
            related_event_id=event_id
        )
        try:
            doc_ref = self.firestore_db.collection(f"users/{user_id}/mezzo_memories").document(record.record_id)
            await asyncio.to_thread(doc_ref.set, record.to_dict())
            logger.info(f"Legacy event '{event_type}' stored as memory for user {user_id}.")
        except Exception as e:
            logger.error(f"Failed to store legacy event as memory for user {user_id}: {e}", exc_info=True)

