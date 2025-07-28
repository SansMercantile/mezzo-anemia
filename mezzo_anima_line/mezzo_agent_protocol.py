# sans-mercantile-app/backend/mezzo_anima_line/mezzo_agent_protocol.py
from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class MezzoAgentType(str, Enum):
    """Defines the types of Mezzo agents."""
    MEZZO_MATERNA = "MEZZO_MATERNA"
    MEZZO_PATERNA = "MEZZO_PATERNA"
    MEZZO_GENERAL = "MEZZO_GENERAL"
    MEZZO_MEMORY_COLLECTOR = "MEZZO_MEMORY_COLLECTOR" # New agent type for deep interviews
    EXTERNAL_API = "EXTERNAL_API"

class MezzoAgentState(str, Enum):
    """Defines the operational states of a Mezzo agent."""
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    PROCESSING_CONVERSATION = "PROCESSING_CONVERSATION"
    PERFORMING_MEMORY_RETRIEVAL = "PERFORMING_MEMORY_RETRIEVAL"
    GENERATING_LEGACY_RECORD = "GENERATING_LEGACY_RECORD"
    COLLECTING_DEEP_MEMORY = "COLLECTING_DEEP_MEMORY" # New state
    PAUSED_MEMORY_COLLECTION = "PAUSED_MEMORY_COLLECTION" # New state
    SHUTTING_DOWN = "SHUTTING_DOWN"
    ERROR = "ERROR"

class MezzoAgentMessage(BaseModel):
    """Standardized message format for inter-Mezzo agent communication."""
    sender_id: str
    sender_type: MezzoAgentType
    recipient_id: str # Can be agent_id or a topic name
    message_type: str # e.g., "user_message", "memory_query", "nurture_request", "legacy_record_task", "start_memory_collection", "continue_memory_collection", "pause_memory_collection", "memory_snippet_collected"
    content: Dict[str, Any] # The actual payload of the message
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    conversation_id: Optional[str] = None # To link messages in a conversation

class MezzoMemoryRecord(BaseModel):
    """Schema for a memory record to be stored in Firestore."""
    record_id: str = Field(default_factory=lambda: f"mem_{datetime.utcnow().timestamp()}")
    user_id: str
    agent_id: str # Which Mezzo agent this memory is associated with
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    type: str # e.g., "conversation_snippet", "emotional_context", "life_event", "preference", "sensory_detail", "childhood_memory", "toddler_memory", "kid_memory", "teen_memory", "adult_memory"
    content: Dict[str, Any] # The actual memory data (e.g., {"text": "...", "emotion": "...", "smell": "...", "friends_count": 5})
    tags: List[str] = [] # For categorization and retrieval
    related_event_id: Optional[str] = None # Link to emotional context or other events
    confidentiality_level: str = "private" # "private", "limited_share", "public"
    can_be_reshared: bool = False # Explicit flag for re-sharing

class MemoryCollectionSession(BaseModel):
    """Tracks the state of a deep memory collection session."""
    session_id: str = Field(default_factory=lambda: f"mem_session_{datetime.utcnow().timestamp()}")
    user_id: str
    mezzo_agent_id: str # The Mezzo agent conducting the session
    current_phase: str # e.g., "birth_to_toddler", "toddler_phase", "kid_phase", "teen_phase", "adult_phase", "current_events"
    last_interaction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    last_question_asked: Optional[str] = None
    progress_details: Dict[str, Any] = {} # e.g., {"childhood_memories_count": 10}
    status: str = "active" # "active", "paused", "completed", "aborted"
    start_timestamp: datetime = Field(default_factory=datetime.utcnow)
    resume_point: Optional[Dict[str, Any]] = None # Where to resume if paused

