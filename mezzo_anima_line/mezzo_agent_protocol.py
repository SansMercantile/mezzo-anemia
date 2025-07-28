# sans-mercantile-app/backend/mezzo_anima_line/mezzo_agent_protocol.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class MezzoAgentType(str, Enum):
    """Defines the types of Mezzo agents."""
    MEZZO_MATERNA = "MEZZO_MATERNA"
    MEZZO_PATERNA = "MEZZO_PATERNA" # Future expansion
    MEZZO_GENERAL = "MEZZO_GENERAL" # Main Mezzo persona
    EXTERNAL_API = "EXTERNAL_API" # For messages originating from outside Mezzo's internal system

class MezzoAgentState(str, Enum):
    """Defines the operational states of a Mezzo agent."""
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    PROCESSING_CONVERSATION = "PROCESSING_CONVERSATION"
    PERFORMING_MEMORY_RETRIEVAL = "PERFORMING_MEMORY_RETRIEVAL"
    GENERATING_LEGACY_RECORD = "GENERATING_LEGACY_RECORD"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    ERROR = "ERROR"

class MezzoAgentMessage(BaseModel):
    """Standardized message format for inter-Mezzo agent communication."""
    sender_id: str
    sender_type: MezzoAgentType
    recipient_id: str # Can be agent_id or a topic name
    message_type: str # e.g., "user_message", "memory_query", "nurture_request", "legacy_record_task"
    content: Dict[str, Any] # The actual payload of the message
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    conversation_id: Optional[str] = None # To link messages in a conversation

class MezzoMemoryRecord(BaseModel):
    """Schema for a memory record to be stored in Firestore."""
    record_id: str = Field(default_factory=lambda: f"mem_{datetime.utcnow().timestamp()}")
    user_id: str
    agent_id: str # Which Mezzo agent this memory is associated with
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    type: str # e.g., "conversation_snippet", "emotional_context", "life_event", "preference"
    content: Dict[str, Any] # The actual memory data (e.g., {"text": "...", "emotion": "..."})
    tags: List[str] = [] # For categorization and retrieval
    related_event_id: Optional[str] = None # Link to emotional context or other events


