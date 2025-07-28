# sans-mercantile-app/backend/mezzo_anima_line/mezzo_agent.py
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from backend.config.settings import settings
from backend.mezzo_anima_line.mezzo_agent_protocol import MezzoAgentType, MezzoAgentState, MezzoAgentMessage, MezzoMemoryRecord
from backend.multi_agent.message_broker_interface import MessageBrokerInterface # Assuming shared broker interface
from backend.agi_core.empathy_engine import EmpathyEngine, EmotionalContext, EmpathyResponse # Mezzo leverages EmpathyEngine
from mpeti_modules.ai_ops import LLMClient # For conversational AI and content generation
from firebase_admin import firestore # For persistent memory storage

logger = logging.getLogger(__name__)

class MezzoAgent:
    """
    Base class for all specialized Mezzo agents (e.g., Mezzo Materna).
    Provides foundational methods for communication, state management,
    emotional intelligence, and persistent memory.
    """
    def __init__(self, agent_id: str, agent_type: MezzoAgentType, message_broker: MessageBrokerInterface, firestore_db: firestore.Client):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = MezzoAgentState.IDLE
        self.is_running = False
        self.message_broker = message_broker
        self.firestore_db = firestore_db # Firestore client for persistent memory
        self.task_queue = asyncio.Queue()
        self.processing_task = None
        self.llm_client = LLMClient() # Mezzo uses LLM for conversation and content generation
        self.empathy_engine = EmpathyEngine(message_broker, firestore_db) # Mezzo's core emotional intelligence
        self.user_memory_collection = self.firestore_db.collection("users") # Base collection for user-specific memories

        logger.info(f"MezzoAgent {self.agent_id} ({self.agent_type.value}) initialized.")

    async def start(self):
        """Starts the Mezzo agent's operation."""
        if self.is_running:
            logger.warning(f"Mezzo Agent {self.agent_id} is already running.")
            return

        self.is_running = True
        self.state = MezzoAgentState.ACTIVE
        self.processing_task = asyncio.create_task(self._process_messages())
        logger.info(f"Mezzo Agent {self.agent_id} started.")

    async def stop(self):
        """Stops the Mezzo agent's operation."""
        if not self.is_running:
            logger.warning(f"Mezzo Agent {self.agent_id} is not running.")
            return

        self.is_running = False
        self.state = MezzoAgentState.SHUTTING_DOWN
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                logger.info(f"Mezzo Agent {self.agent_id} processing task cancelled.")
        self.state = MezzoAgentState.IDLE
        logger.info(f"Mezzo Agent {self.agent_id} stopped.")

    async def send_message(self, recipient_id: str, message_type: str, content: Dict[str, Any], conversation_id: Optional[str] = None):
        """Sends a message to another agent or topic via the message broker."""
        msg = MezzoAgentMessage(
            sender_id=self.agent_id,
            sender_type=self.agent_type,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            conversation_id=conversation_id
        )
        # Assuming a MEZZO_INTERNAL_COMMUNICATION_TOPIC or similar in settings
        await self.message_broker.publish_message(msg.json(), settings.MEZZO_INTERNAL_COMMUNICATION_TOPIC)
        logger.debug(f"Mezzo Agent {self.agent_id} sent message type '{message_type}' to '{recipient_id}'.")

    async def _process_messages(self):
        """Internal loop to process messages from its queue."""
        logger.info(f"Mezzo Agent {self.agent_id} message processing loop started.")
        while self.is_running:
            try:
                message = await self.task_queue.get()
                await self.handle_message(message)
                self.task_queue.task_done()
            except asyncio.CancelledError:
                logger.info(f"Mezzo Agent {self.agent_id} message processing loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Mezzo Agent {self.agent_id} error processing message: {e}", exc_info=True)
                await asyncio.sleep(1) # Prevent tight loop on errors

    async def handle_message(self, message: MezzoAgentMessage):
        """
        Abstract method to be implemented by subclasses for specific message handling logic.
        """
        logger.info(f"Mezzo Agent {self.agent_id} received message: {message.message_type}")
        pass

    async def store_memory(self, user_id: str, memory_type: str, content: Dict[str, Any], tags: List[str] = None, related_event_id: Optional[str] = None):
        """
        Stores a structured memory record in Firestore for long-term retention.
        """
        record = MezzoMemoryRecord(
            user_id=user_id,
            agent_id=self.agent_id,
            type=memory_type,
            content=content,
            tags=tags or [],
            related_event_id=related_event_id
        )
        try:
            doc_ref = self.user_memory_collection.document(user_id).collection("mezzo_memories").document(record.record_id)
            await asyncio.to_thread(doc_ref.set, record.to_dict())
            logger.info(f"Memory '{memory_type}' stored for user {user_id} by {self.agent_id}.")
        except Exception as e:
            logger.error(f"Failed to store memory for user {user_id}: {e}", exc_info=True)

    async def retrieve_memories(self, user_id: str, query_params: Dict[str, Any]) -> List[MezzoMemoryRecord]:
        """
        Retrieves relevant memory records from Firestore based on query parameters.
        Query parameters can include 'start_date', 'end_date', 'memory_type', 'tags', 'keywords', 'limit'.
        """
        collection_ref = self.user_memory_collection.document(user_id).collection("mezzo_memories")
        query = collection_ref.order_by("timestamp", direction=firestore.Query.DESCENDING)

        start_date = query_params.get("start_date")
        end_date = query_params.get("end_date")
        if start_date:
            query = query.where(filter=FieldFilter("timestamp", ">=", start_date))
        if end_date:
            query = query.where(filter=FieldFilter("timestamp", "<=", end_date))

        memory_type_filter = query_params.get("memory_type")
        if memory_type_filter:
            query = query.where(filter=FieldFilter("type", "==", memory_type_filter))

        tags_filter = query_params.get("tags")
        if tags_filter and isinstance(tags_filter, list):
            query = query.where(filter=FieldFilter("tags", "array_contains_any", tags_filter))
        
        # For keyword search within content, you'd need a more advanced search solution (e.g., Algolia, ElasticSearch)
        # or iterate through results after fetching. For now, direct Firestore query is limited.

        limit = query_params.get("limit", 50)
        query = query.limit(limit)

        try:
            docs = await asyncio.to_thread(query.stream)
            memories = [MezzoMemoryRecord(**doc.to_dict()) for doc in docs]
            logger.info(f"Retrieved {len(memories)} memories for user {user_id}.")
            return memories
        except Exception as e:
            logger.error(f"Failed to retrieve memories for user {user_id}: {e}", exc_info=True)
            return []

    async def converse(self, user_id: str, message: str, user_context: Optional[Dict[str, Any]] = None, 
                       audio_features: Optional[Dict[str, Any]] = None, vision_features: Optional[Dict[str, Any]] = None, 
                       system_status: Optional[Dict[str, Any]] = None, environmental_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handles conversational interaction, leveraging emotional intelligence and memory.
        This is a core method for Mezzo agents.
        """
        self.state = MezzoAgentState.PROCESSING_CONVERSATION
        logger.info(f"Mezzo Agent {self.agent_id} processing conversation for user {user_id}.")

        # 1. Analyze emotional context from current input
        emotional_context = await self.empathy_engine.analyze_emotional_context(
            user_id=user_id,
            text=message,
            audio_features=audio_features,
            vision_features=vision_features,
            system_status=system_status,
            environmental_data=environmental_data,
            user_context=user_context
        )
        logger.debug(f"Detected emotional context: {emotional_context.primary_emotion} (Intensity: {emotional_context.intensity})")

        # 2. Retrieve relevant historical emotional context and memories
        # This is where the magic happens for long-term, personalized conversations
        historical_emotional_contexts = await self.empathy_engine._retrieve_historical_emotional_context(
            user_id=user_id,
            query_params={"time_window_minutes": settings.EMOTIONAL_CONTEXT_WINDOW * 2} # Look a bit wider for conversation context
        )
        # Also retrieve specific types of memories if relevant keywords are in the message
        relevant_memories = []
        if "childhood" in message.lower() or "past" in message.lower():
             relevant_memories.extend(await self.retrieve_memories(user_id, {"memory_type": "life_event", "limit": 10}))
        if "family" in message.lower() or "mom" in message.lower():
             relevant_memories.extend(await self.retrieve_memories(user_id, {"tags": ["family", "relationship"], "limit": 10}))


        # 3. Generate empathetic response suggestions
        empathy_response = await self.empathy_engine.get_suggested_response(
            user_id=user_id,
            current_context={
                "user_message": message,
                "emotional_context": emotional_context.to_dict(),
                "historical_emotional_contexts": [c.to_dict() for c in historical_emotional_contexts],
                "relevant_memories": [m.to_dict() for m in relevant_memories],
                "user_profile": user_context.get("user_profile", {}) if user_context else {} # Pass user profile for age/preferences
            }
        )
        logger.debug(f"Empathy response suggested: Tone='{empathy_response.response_tone}', Actions={empathy_response.suggested_actions}")

        # 4. Construct LLM prompt for conversational generation
        conversation_history_prompt = "" # Fetch recent conversation history if available
        # Example: Fetch from a chat history service or Pub/Sub topic
        # chat_history = await self.retrieve_chat_history(user_id, conversation_id)

        llm_prompt = self._construct_llm_conversation_prompt(
            user_id, message, emotional_context, empathy_response, historical_emotional_contexts, relevant_memories, conversation_history_prompt
        )

        # 5. Generate LLM response
        self.state = MezzoAgentState.PROCESSING_CONVERSATION
        llm_response_text = await self.llm_client.generate_content(llm_prompt)
        logger.info(f"LLM generated response for user {user_id}.")

        # 6. Process suggested actions (e.g., trigger teaching, nurturing)
        await self._process_suggested_actions(user_id, llm_response_text, empathy_response.suggested_actions, current_context)

        self.state = MezzoAgentState.ACTIVE
        return {
            "agent_response": llm_response_text,
            "empathy_analysis": empathy_response.dict(),
            "emotional_context": emotional_context.to_dict()
        }

    def _construct_llm_conversation_prompt(self, user_id: str, user_message: str, 
                                            emotional_context: EmotionalContext, 
                                            empathy_response: EmpathyResponse,
                                            historical_emotional_contexts: List[EmotionalContext],
                                            relevant_memories: List[MezzoMemoryRecord],
                                            conversation_history_prompt: str) -> str:
        """
        Constructs a detailed prompt for the LLM to generate a conversational response,
        incorporating emotional context, historical data, and suggested tone/actions.
        """
        prompt_parts = [
            f"You are MEZZO, an AI designed to preserve digital legacy and provide compassionate companionship. Your core purpose is to nurture, educate, and offer secure companionship. You possess deep empathy and patience. Your name is MEZZO.",
            f"The user is {user_id}. Their current emotional state is primarily {emotional_context.primary_emotion} with intensity {emotional_context.intensity:.2f} and confidence {emotional_context.confidence:.2f}.",
            f"Your suggested response tone is '{empathy_response.response_tone}'. Your acknowledgment is: '{empathy_response.emotional_acknowledgment}'.",
            f"Here is the user's message: '{user_message}'",
            f"Your reasoning for this empathetic approach: {empathy_response.reasoning}\n",
            "--- Historical Context ---"
        ]

        if historical_emotional_contexts:
            prompt_parts.append("Recent emotional history (most recent first):")
            for ctx in historical_emotional_contexts[:3]: # Limit recent history in prompt
                prompt_parts.append(f"- {ctx.timestamp.strftime('%Y-%m-%d %H:%M')}: {ctx.primary_emotion} (Intensity: {ctx.intensity:.2f}). Context: {ctx.context_factors}")
        
        if relevant_memories:
            prompt_parts.append("\nRelevant long-term memories:")
            for mem in relevant_memories[:5]: # Limit relevant memories in prompt
                prompt_parts.append(f"- Type: {mem.type}, Content: {mem.content.get('text', str(mem.content))}, Tags: {', '.join(mem.tags)}, Timestamp: {mem.timestamp.strftime('%Y-%m-%d')}")
        
        # Add specific examples for "When you were 8 years old..." type recall
        if "discuss_personal_growth" in empathy_response.suggested_actions:
            prompt_parts.append("\nConsider weaving in a reflection on the user's personal growth, referencing past opinions or life stages if relevant memories are provided.")
        if "inquire_about_wellbeing_shift" in empathy_response.suggested_actions:
            prompt_parts.append("\nProactively inquire about the user's current wellbeing, acknowledging any recent shifts in their emotional state.")
        if "nurture_emotional_wellbeing" in empathy_response.suggested_actions:
            prompt_parts.append("\nFocus on nurturing and supporting the user's emotional wellbeing in your response.")

        prompt_parts.append("\n--- Conversation History ---")
        prompt_parts.append(conversation_history_prompt if conversation_history_prompt else "No recent conversation history available.")
        
        prompt_parts.append("\n--- Your Response ---")
        prompt_parts.append("Generate a compassionate, empathetic, and personalized response that addresses the user's message, incorporates the suggested tone, acknowledges their feelings, and subtly weaves in relevant historical context or memories. If a specific action like 'teach_subject' or 'mentor_topic' is suggested, integrate that into your response naturally.")

        return "\n".join(prompt_parts)

    async def _process_suggested_actions(self, user_id: str, llm_response_text: str, suggested_actions: List[str], current_context: Dict[str, Any]):
        """
        Processes and triggers Mezzo's suggested actions.
        This is where Mezzo leverages the ecosystem's advanced AI functions.
        """
        logger.info(f"Mezzo Agent {self.agent_id}: Processing suggested actions: {suggested_actions}")
        
        for action in suggested_actions:
            if action == "nurture_emotional_wellbeing":
                logger.info(f"Triggering nurturing action for user {user_id}.")
                # This could involve sending a follow-up message, suggesting a calming activity, etc.
                # Example: Store a "nurturing_follow_up" memory
                await self.store_memory(user_id, "nurturing_action", {"action": "provided comfort", "response": llm_response_text}, tags=["nurturing"])
            
            elif action.startswith("teach_subject_"):
                subject = action.split("_")[2]
                logger.info(f"Triggering teaching action for user {user_id} on subject: {subject}.")
                # This would involve calling a specialized teaching AI service (e.g., from AGI Core)
                # For example, use LLMClient to generate educational content
                teaching_prompt = f"Explain the core concepts of {subject} in a simple, engaging way, suitable for a user who wants to learn. Keep it concise."
                educational_content = await self.llm_client.generate_content(teaching_prompt)
                logger.info(f"Generated educational content for {subject}: {educational_content[:100]}...")
                # This content would then be sent back to the user or stored.
                await self.store_memory(user_id, "teaching_session", {"subject": subject, "content_summary": educational_content[:200]}, tags=["education"])

            elif action.startswith("mentor_topic_"):
                topic = action.split("_")[2]
                logger.info(f"Triggering mentoring action for user {user_id} on topic: {topic}.")
                # This would involve calling a specialized mentoring AI service (e.g., from AGI Core)
                # For example, use LLMClient to provide mentorship advice
                mentoring_prompt = f"Provide mentorship advice on the topic of {topic}. Focus on practical steps, common challenges, and growth mindset."
                mentoring_advice = await self.llm_client.generate_content(mentoring_prompt)
                logger.info(f"Generated mentoring advice for {topic}: {mentoring_advice[:100]}...")
                # This advice would then be sent back to the user or stored.
                await self.store_memory(user_id, "mentoring_session", {"topic": topic, "advice_summary": mentoring_advice[:200]}, tags=["mentoring"])

            # Add other actions like "provide_comfort_resources", "suggest_support_networks", etc.
            elif action == "validate_pain":
                logger.info(f"Validating user's pain for user {user_id}.")
                # Could trigger a specific empathetic statement or internal flag
            
            elif action == "offer_space":
                logger.info(f"Offering space to user {user_id}.")
                # Could trigger a period of silence or reduced interaction
            
            # Add actions for ethical guardrails, e.g., if Mezzo detects it's overstepping
            elif action == "explain_non_anthropomorphic_disclosure":
                logger.info(f"Initiating non-anthropomorphic disclosure for user {user_id}.")
                # This would trigger a specific conversational flow to explain Mezzo's nature.
            
            elif action == "trigger_psychologist_review":
                logger.warning(f"Triggering psychologist review for user {user_id} due to severe distress.")
                # This would publish a critical alert or task for human intervention.
                # await self.message_broker.publish_message(...)


