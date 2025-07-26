# backend/mezzo_anima_line/dependencies.py

import logging
from .config import settings
from .digital_legacy_manager import DigitalLegacyManager
from .grief_dialogue_engine import GriefDialogueEngine
from .emotional_sentinel import EmotionalSentinel
from backend.support_ai.core_ai_handler import LLMClient
from backend.communication.notification_manager import NotificationManager

logger = logging.getLogger(__name__)

_singletons = {}
_firestore_db = None

def set_firestore_db(db_client):
    """Sets the global Firestore client for this application's context."""
    global _firestore_db
    _firestore_db = db_client

def get_firestore_db():
    """Returns the initialized Firestore client."""
    if not _firestore_db:
        raise RuntimeError("Firestore has not been initialized for the MEZZO application.")
    return _firestore_db

def get_singleton(name: str):
    """A helper to get any initialized singleton by name."""
    return _singletons.get(name)

def get_llm_client() -> LLMClient:
    if "llm_client" not in _singletons:
        _singletons["llm_client"] = LLMClient()
    return _singletons["llm_client"]

def get_notification_manager() -> NotificationManager:
    if "notification_manager" not in _singletons:
        _singletons["notification_manager"] = NotificationManager()
    return _singletons["notification_manager"]

def get_digital_legacy_manager() -> DigitalLegacyManager:
    if "digital_legacy_manager" not in _singletons:
        _singletons["digital_legacy_manager"] = DigitalLegacyManager()
    return _singletons["digital_legacy_manager"]

def get_grief_dialogue_engine() -> GriefDialogueEngine:
    if "grief_dialogue_engine" not in _singletons:
        llm_client = get_llm_client()
        _singletons["grief_dialogue_engine"] = GriefDialogueEngine(llm_client)
    return _singletons["grief_dialogue_engine"]

def get_emotional_sentinel() -> EmotionalSentinel:
    if "emotional_sentinel" not in _singletons:
        _singletons["emotional_sentinel"] = EmotionalSentinel()
    return _singletons["emotional_sentinel"]

def initialize_mezzo_singletons():
    """
    Initializes all singleton services required for the MEZZO application.
    """
    global _singletons
    if "initialized" in _singletons:
        return

    logger.info("Initializing all MEZZO-specific singleton services...")
    
    _singletons["llm_client"] = get_llm_client()
    _singletons["notification_manager"] = get_notification_manager()
    _singletons["digital_legacy_manager"] = get_digital_legacy_manager()
    _singletons["grief_dialogue_engine"] = get_grief_dialogue_engine()
    _singletons["emotional_sentinel"] = get_emotional_sentinel()

    _singletons["initialized"] = True
    logger.info("All MEZZO singleton services have been initialized.")

