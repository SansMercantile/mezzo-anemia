# backend/mezzo_anima_line/anima_engine.py

import logging
from typing import List, Dict, Any
from .memory_ingestion import Memory, MemoryIngestionPipeline
from .persona_simulation import Persona

logger = logging.getLogger(__name__)

class AnimaEngine:
    """
    The main engine for the Mezzo "Anima Line". It orchestrates the ingestion
    of memories and the creation and interaction with a simulated Persona.
    """
    def __init__(self, persona_name: str):
        self.persona_name = persona_name
        self.ingestion_pipeline = MemoryIngestionPipeline()
        self.memories: List[Memory] = []
        self.persona: Persona | None = None
        logger.info(f"AnimaEngine for persona '{self.persona_name}' created.")

    def ingest_memory(self, raw_text: str, source_type: str, author: str, timestamp: str):
        """
        Ingests a new piece of raw data, processes it, and adds it to the memory bank.
        """
        from datetime import datetime
        dt_object = datetime.fromisoformat(timestamp)
        new_memory = self.ingestion_pipeline.process_text(
            raw_text=raw_text,
            source_type=source_type,
            author=author,
            timestamp=dt_object
        )
        self.memories.append(new_memory)
        logger.info(f"New memory (ID: {new_memory.id}) ingested. Total memories: {len(self.memories)}.")
        # After ingesting new memories, the persona should be rebuilt to incorporate them.
        self.rebuild_persona()

    def rebuild_persona(self):
        """
        Re-creates the Persona object using the current set of memories.
        This should be called after new memories are added.
        """
        if not self.memories:
            logger.warning("Cannot build persona, no memories have been ingested.")
            return
        self.persona = Persona(persona_name=self.persona_name, memories=self.memories)
        logger.info(f"Persona '{self.persona_name}' has been (re)built.")

    def interact(self, prompt: str) -> str:
        """
        Interacts with the simulated persona.
        """
        if not self.persona:
            return "The persona has not been built yet. Please ingest memories first."
        
        return self.persona.generate_response(prompt)

