# backend/mezzo_anima_line/digital_legacy_manager.py

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

class DigitalLegacyManager:
    """
    Manages the secure storage, access control, and release of digital legacy assets.
    """
    def __init__(self, encryption_key: bytes):
        self.assets: Dict[str, Dict[str, Any]] = {}
        self.cipher_suite = Fernet(encryption_key)

    def ingest_document(self, user_id: str, document_name: str, document_content: str, release_conditions: Dict[str, Any]):
        """
        Ingests and encrypts a legal document or personal message.
        """
        encrypted_content = self.cipher_suite.encrypt(document_content.encode())
        asset_id = f"{user_id}_{document_name}"
        self.assets[asset_id] = {
            "user_id": user_id,
            "document_name": document_name,
            "encrypted_content": encrypted_content,
            "release_conditions": release_conditions,
            "is_released": False
        }
        logger.info(f"Ingested and encrypted document '{document_name}' for user {user_id}.")

    def grant_access(self, asset_id: str, beneficiary_id: str, access_level: str = "read"):
        """
        Grants a beneficiary access to a specific digital asset.
        """
        if asset_id in self.assets:
            if "beneficiaries" not in self.assets[asset_id]:
                self.assets[asset_id]["beneficiaries"] = []
            self.assets[asset_id]["beneficiaries"].append({
                "beneficiary_id": beneficiary_id,
                "access_level": access_level
            })
            logger.info(f"Granted access for beneficiary {beneficiary_id} to asset {asset_id}.")

    def check_and_release_assets(self) -> List[str]:
        """
        Checks release conditions and releases assets that meet their criteria.
        """
        released_assets = []
        for asset_id, asset in self.assets.items():
            if not asset["is_released"]:
                # In a real system, this would involve complex condition checking
                # (e.g., verifying a death certificate, checking a specific date)
                if asset["release_conditions"].get("release_on_date") == datetime.utcnow().strftime("%Y-%m-%d"):
                    asset["is_released"] = True
                    released_assets.append(asset_id)
                    logger.info(f"Released asset {asset_id} based on release conditions.")
        return released_assets
