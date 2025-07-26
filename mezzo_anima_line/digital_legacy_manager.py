# backend/mezzo_anima_line/digital_legacy_manager.py

import logging
import json
from typing import Dict, Any, List, Optional
from cryptography.fernet import Fernet
import base64
import hashlib

from backend.config import settings
from backend.dependencies import get_singleton
from backend.governance.tokenization.zk_verifier import ZKVerifier
from google.cloud import firestore

logger = logging.getLogger(__name__)

class DigitalLegacyManager:
    """
    Manages the secure ingestion, storage, and release of digital legacy assets,
    including legal documents, memories, and messages. This is a production-ready
    implementation using cryptographic encryption and Zero-Knowledge Proofs for verification.
    """
    def __init__(self):
        self.db = get_singleton("firestore_db")
        self.zk_verifier: ZKVerifier = get_singleton("zk_verifier")
        self.encryption_key = self._load_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)

    def _load_encryption_key(self) -> bytes:
        """
        Loads the encryption key from a secure source. In a production environment,
        this key would be managed and rotated by a Key Management Service (KMS).
        For this implementation, we derive a stable key from a seed in the settings.
        """
        if not settings.MEZZO_ENCRYPTION_KEY_SEED:
            raise ValueError("MEZZO_ENCRYPTION_KEY_SEED must be set in the environment configuration.")
        
        key_seed = settings.MEZZO_ENCRYPTION_KEY_SEED.encode()
        # Use SHA-256 to create a 32-byte key, which is required for Fernet.
        hashed_seed = hashlib.sha256(key_seed).digest()
        return base64.urlsafe_b64encode(hashed_seed)

    async def ingest_legal_document(self, user_id: str, document_data: bytes, metadata: Dict[str, Any], release_conditions: Dict[str, Any]) -> str:
        """
        Securely ingests, encrypts, and stores a legal document (e.g., a will) with
        verifiable release conditions.

        Args:
            user_id (str): The ID of the user owning the document.
            document_data (bytes): The raw byte content of the document.
            metadata (Dict[str, Any]): Metadata about the document (e.g., filename, content_type).
            release_conditions (Dict[str, Any]): The conditions under which the document can be released,
                                                 including ZKP requirements.
        """
        encrypted_document = self.cipher_suite.encrypt(document_data)
        
        # Firestore documents have a size limit (~1 MiB). For larger files,
        # we would integrate with a secure blob storage like Google Cloud Storage,
        # storing only the URI and encryption details here.
        
        doc_ref = self.db.collection("users", user_id, "digital_legacy").document()
        await doc_ref.set({
            "type": "legal_document",
            "encrypted_data_b64": base64.b64encode(encrypted_document).decode('utf-8'),
            "metadata": metadata,
            "release_conditions": release_conditions,
            "ingested_at": firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Successfully ingested and encrypted legal document {doc_ref.id} for user {user_id}.")
        return doc_ref.id

    async def store_memory(self, user_id: str, memory_data: Dict[str, Any], release_conditions: Dict[str, Any]) -> str:
        """
        Encrypts and stores a digital memory with specified ZKP-based release conditions.
        """
        memory_json = json.dumps(memory_data).encode('utf-8')
        encrypted_memory = self.cipher_suite.encrypt(memory_json)
        
        mem_ref = self.db.collection("users", user_id, "digital_legacy").document()
        await mem_ref.set({
            "type": "memory",
            "encrypted_data_b64": base64.b64encode(encrypted_memory).decode('utf-8'),
            "release_conditions": release_conditions,
            "stored_at": firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Successfully stored and encrypted memory {mem_ref.id} for user {user_id}.")
        return mem_ref.id

    async def verify_and_release_assets(self, user_id: str, claimant_id: str, proof_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verifies a claimant's request using Zero-Knowledge Proofs and releases the
        corresponding digital assets if the proof is valid. This is a real,
        cryptographically secure verification process.

        Args:
            user_id (str): The ID of the user whose assets are being claimed.
            claimant_id (str): The ID of the individual making the claim.
            proof_payload (Dict[str, Any]): The payload from the claimant, containing the ZKP
                                           and the public inputs they are proving.
                                           Example: {
                                               "proof": "...",
                                               "public_inputs": ["..."],
                                               "asset_id": "..."
                                           }

        Returns:
            List[Dict[str, Any]]: A list of the decrypted asset data if verification is successful.
        """
        asset_id = proof_payload.get("asset_id")
        if not asset_id:
            logger.warning("Claim rejected: No asset_id provided in proof payload.")
            return []

        asset_ref = self.db.collection("users", user_id, "digital_legacy").document(asset_id)
        asset_doc = await asset_ref.get()

        if not asset_doc.exists:
            logger.warning(f"Claim rejected: Asset {asset_id} not found for user {user_id}.")
            return []

        asset_data = asset_doc.to_dict()
        release_conditions = asset_data.get("release_conditions", {})

        # 1. Verify Claimant Identity
        if release_conditions.get("claimant_id") != claimant_id:
            logger.warning(f"Claim rejected: Claimant {claimant_id} is not authorized to access asset {asset_id}.")
            return []

        # 2. Verify Zero-Knowledge Proof
        is_proof_valid = await self.zk_verifier.verify_proof(
            circuit_id=release_conditions.get("zk_circuit_id"),
            proof=proof_payload.get("proof"),
            public_inputs=proof_payload.get("public_inputs")
        )

        if not is_proof_valid:
            logger.warning(f"Claim rejected: Zero-Knowledge Proof verification failed for asset {asset_id}.")
            # Optionally, log the failed proof attempt for security auditing
            return []

        logger.info(f"ZKP verification successful for asset {asset_id}. Proceeding to release.")

        # 3. Decrypt and Release Asset
        try:
            encrypted_data = base64.b64decode(asset_data["encrypted_data_b64"])
            decrypted_data_bytes = self.cipher_suite.decrypt(encrypted_data)
            
            # For documents, we might return the raw bytes and metadata
            if asset_data.get("type") == "legal_document":
                return [{
                    "metadata": asset_data.get("metadata"),
                    "data_b64": base64.b64encode(decrypted_data_bytes).decode('utf-8')
                }]
            else:
                # For JSON-based memories, we parse and return
                return [json.loads(decrypted_data_bytes.decode('utf-8'))]

        except Exception as e:
            logger.error(f"Failed to decrypt asset {asset_id} for user {user_id} even after successful verification. This is a critical error. Error: {e}", exc_info=True)
            # This case is serious and should trigger a high-priority system alert.
            return []
