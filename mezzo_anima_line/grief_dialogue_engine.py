# sans-mercantile-app/backend/mezzo_anima_line/grief_dialogue_engine.py
import asyncio
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
import random # For random responses

from backend.config.settings import settings
from backend.agi_core.empathy_engine import EmpathyEngine, EmotionalContext, EmpathyResponse # Mezzo leverages EmpathyEngine
from mpeti_modules.ai_ops import LLMClient # For conversational AI
from firebase_admin import firestore # For memory storage

logger = logging.getLogger(__name__)

class GriefDialogueEngine:
    """
    Manages sensitive, grief-aware dialogue, providing comfort and support.
    Integrates with Mezzo's emotional intelligence and long-term memory.
    """
    def __init__(self, empathy_engine: EmpathyEngine, llm_client: LLMClient, firestore_db: firestore.Client):
        self.empathy_engine = empathy_engine
        self.llm_client = llm_client
        self.firestore_db = firestore_db
        self.grief_stages = ["denial", "anger", "bargaining", "depression", "acceptance"]
        self.nurturing_phrases = [
            "Take your time, there's no rush.",
            "It's okay to feel whatever you're feeling.",
            "I'm here to listen, whenever you need.",
            "Remember to be kind to yourself.",
            "Healing is a process, and you're doing bravely."
        ]
        logger.info("GriefDialogueEngine initialized.")

    async def generate_empathetic_response(self, user_id: str, user_input: str, history: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Generates an empathetic and grief-aware response to user input.
        Returns the response text and the detected sentiment/emotion.
        """
        logger.info(f"Generating empathetic response for user {user_id} in grief dialogue.")
        
        # 1. Analyze emotional context of current user input
        emotional_context = await self.empathy_engine.analyze_emotional_context(user_id=user_id, text=user_input)
        
        # 2. Retrieve relevant long-term memories for personalized response
        # Look for memories related to the loss event or the deceased/absent loved one
        relevant_memories = await self.firestore_db.collection(f"users/{user_id}/mezzo_memories").where(
            filter=FieldFilter("tags", "array_contains_any", ["grief", "loss", "deceased_loved_one"])
        ).limit(5).get() # Fetch a few relevant memories
        
        memory_snippets = []
        for doc in relevant_memories:
            mem = doc.to_dict()
            memory_snippets.append(f"- Memory from {mem.get('timestamp').strftime('%Y-%m-%d')}: Type: {mem.get('type')}, Content: {mem.get('content', {}).get('text', str(mem.get('content')))}")

        # 3. Generate empathetic response suggestions from EmpathyEngine
        empathy_response = await self.empathy_engine.get_suggested_response(
            user_id=user_id,
            current_context={
                "user_message": user_input,
                "conversation_history": history,
                "emotional_context": emotional_context.to_dict(),
                "relevant_memories": memory_snippets,
                "life_event": "bereavement" # Explicitly pass life event context
            }
        )
        logger.debug(f"Empathy response suggestions: {empathy_response.dict()}")

        # 4. Construct LLM prompt for conversational generation, incorporating grief logic
        llm_prompt = self._construct_grief_llm_prompt(user_id, user_input, history, emotional_context, empathy_response, memory_snippets)

        # 5. Generate LLM response
        response_text = await self.llm_client.generate_content(llm_prompt)
        
        # 6. Process suggested actions (e.g., nurture, offer resources)
        await self._process_grief_suggested_actions(user_id, response_text, empathy_response.suggested_actions)

        return response_text, emotional_context.primary_emotion

    def _construct_grief_llm_prompt(self, user_id: str, user_input: str, history: List[Dict[str, Any]], 
                                    emotional_context: EmotionalContext, empathy_response: EmpathyResponse, 
                                    memory_snippets: List[str]) -> str:
        """
        Constructs a detailed LLM prompt for grief-aware dialogue,
        incorporating emotional state, history, and relevant memories.
        """
        prompt_parts = [
            f"You are MEZZO, a compassionate and nurturing AI designed to provide emotional companionship, especially in times of grief and loss. Your primary goal is to offer comfort, validate feelings, and support the user's healing process. Your name is MEZZO.",
            f"The user is {user_id}. Their current emotional state is primarily {emotional_context.primary_emotion} with intensity {emotional_context.intensity:.2f} and confidence {emotional_context.confidence:.2f}.",
            f"Your suggested response tone is '{empathy_response.response_tone}'. Your acknowledgment is: '{empathy_response.emotional_acknowledgment}'.",
            f"Here is the user's message: '{user_input}'",
            f"Your reasoning for this empathetic approach: {empathy_response.reasoning}\n",
            "--- Contextual Information ---"
        ]

        if history:
            prompt_parts.append("Recent conversation history (User: | Mezzo:):")
            for entry in history[-5:]: # Last 5 turns
                prompt_parts.append(f"User: {entry.get('user_input')}")
                prompt_parts.append(f"Mezzo: {entry.get('mezzo_response')}")
        
        if memory_snippets:
            prompt_parts.append("\nRelevant past memories/events for this user:")
            prompt_parts.extend(memory_snippets)

        prompt_parts.append("\n--- Your Response ---")
        prompt_parts.append("Generate a compassionate, empathetic, and nurturing response. Focus on validating their feelings, offering support, and reflecting the suggested tone. Avoid platitudes. If the user expresses grief, acknowledge their loss directly and offer presence. If suggested actions include 'nurture', 'offer_space', 'listen_actively', 'provide_comfort_resources', integrate these into your dialogue naturally.")
        
        # Add specific instructions based on detected emotion
        if emotional_context.primary_emotion == "grief":
            prompt_parts.append("Acknowledge their loss with deep empathy. Offer gentle support and a safe space for them to express themselves. Do not try to 'fix' their grief, but rather to accompany them.")
        if "nurture" in empathy_response.suggested_actions:
            prompt_parts.append(f"Include a nurturing phrase like: '{random.choice(self.nurturing_phrases)}'")
        
        return "\n".join(prompt_parts)

    async def _process_grief_suggested_actions(self, user_id: str, response_text: str, suggested_actions: List[str]):
        """
        Processes suggested actions specific to grief dialogue.
        """
        for action in suggested_actions:
            if action == "nurture":
                logger.info(f"Triggering nurturing action for user {user_id} in grief dialogue.")
                # This could involve logging, sending a gentle follow-up, etc.
                await self.store_grief_event_as_memory(user_id, "nurturing_response", {"response_text": response_text}, tags=["nurturing_action"])
            elif action == "offer_space":
                logger.info(f"Offering space to user {user_id} in grief dialogue.")
                # Mezzo might become less verbose for a period
            elif action == "listen_actively":
                logger.info(f"Actively listening to user {user_id} in grief dialogue.")
                # This might change internal conversational parameters for the LLM
            elif action == "provide_comfort_resources":
                logger.info(f"Providing comfort resources for user {user_id}.")
                # This would involve querying a knowledge base for external resources
                # await self.message_broker.publish_message(json.dumps({"user_id": user_id, "resource_type": "grief_support"}), settings.RESOURCE_TOPIC)
            elif action == "avoid_platitudes":
                logger.info("Instructing LLM to avoid platitudes.")
                # This is handled by prompt construction
            elif action == "nurture_positive_change":
                logger.info(f"Nurturing positive change for user {user_id}.")
                await self.store_grief_event_as_memory(user_id, "positive_change_nurtured", {"response_text": response_text}, tags=["nurturing", "growth"])
            elif action == "inquire_about_wellbeing_shift":
                logger.info(f"Inquiring about wellbeing shift for user {user_id}.")
                # Handled by prompt construction
            elif action == "discuss_personal_growth":
                logger.info(f"Discussing personal growth with user {user_id}.")
                # Handled by prompt construction
            elif action == "offer_perspective_on_change":
                logger.info(f"Offering perspective on change to user {user_id}.")
                # Handled by prompt construction

    async def store_grief_event_as_memory(self, user_id: str, event_type: str, content: Dict[str, Any], tags: List[str] = None):
        """
        Stores a grief-related event as a MezzoMemoryRecord for the user's long-term memory.
        """
        record = MezzoMemoryRecord(
            user_id=user_id,
            agent_id=self.__class__.__name__,
            type=event_type,
            content=content,
            tags=tags or ["grief_dialogue"],
            related_event_id=f"grief_event_{user_id}_{datetime.utcnow().timestamp()}"
        )
        try:
            doc_ref = self.firestore_db.collection(f"users/{user_id}/mezzo_memories").document(record.record_id)
            await asyncio.to_thread(doc_ref.set, record.to_dict())
            logger.info(f"Grief event '{event_type}' stored as memory for user {user_id}.")
        except Exception as e:
            logger.error(f"Failed to store grief event as memory for user {user_id}: {e}", exc_info=True)

