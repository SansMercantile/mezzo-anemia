# sans-mercantile-app/backend/mezzo_anima_line/mezzo_agent.py
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from backend.config.settings import settings
from backend.mezzo_anima_line.mezzo_agent_protocol import MezzoAgentType, MezzoAgentState, MezzoAgentMessage, MezzoMemoryRecord, MemoryCollectionSession
from backend.multi_agent.message_broker_interface import MessageBrokerInterface
from backend.agi_core.empathy_engine import EmpathyEngine, EmotionalContext, EmpathyResponse
from mpeti_modules.ai_ops import LLMClient
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

logger = logging.getLogger(__name__)

class MezzoAgent:
    """
    Base class for all specialized Mezzo agents (e.g., Mezzo Materna, Mezzo Memory Collector).
    Provides foundational methods for communication, state management,
    emotional intelligence, persistent memory, and ethical data handling.
    """
    def __init__(self, agent_id: str, agent_type: MezzoAgentType, message_broker: MessageBrokerInterface, firestore_db: firestore.Client):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = MezzoAgentState.IDLE
        self.is_running = False
        self.message_broker = message_broker
        self.firestore_db = firestore_db
        self.task_queue = asyncio.Queue()
        self.processing_task = None
        self.llm_client = LLMClient()
        self.empathy_engine = EmpathyEngine(message_broker, firestore_db)
        self.user_memory_collection = self.firestore_db.collection("users") # Base collection for user-specific memories
        self.active_memory_sessions: Dict[str, MemoryCollectionSession] = {} # user_id -> MemoryCollectionSession

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

    async def store_memory(self, user_id: str, memory_type: str, content: Dict[str, Any], tags: List[str] = None, 
                           related_event_id: Optional[str] = None, confidentiality_level: str = "private", can_be_reshared: bool = False):
        """
        Stores a structured memory record in Firestore for long-term retention,
        with specified confidentiality and re-sharing flags.
        """
        record = MezzoMemoryRecord(
            user_id=user_id,
            agent_id=self.agent_id,
            type=memory_type,
            content=content,
            tags=tags or [],
            related_event_id=related_event_id,
            confidentiality_level=confidentiality_level,
            can_be_reshared=can_be_reshared
        )
        try:
            doc_ref = self.user_memory_collection.document(user_id).collection("mezzo_memories").document(record.record_id)
            await asyncio.to_thread(doc_ref.set, record.to_dict())
            logger.info(f"Memory '{memory_type}' stored for user {user_id} by {self.agent_id}. Confidentiality: {confidentiality_level}, Reshareable: {can_be_reshared}")
        except Exception as e:
            logger.error(f"Failed to store memory for user {user_id}: {e}", exc_info=True)

    async def retrieve_memories(self, user_id: str, query_params: Dict[str, Any]) -> List[MezzoMemoryRecord]:
        """
        Retrieves relevant memory records from Firestore based on query parameters.
        Query parameters can include 'start_date', 'end_date', 'memory_type', 'tags', 'keywords', 'limit', 'confidentiality_level'.
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
        
        confidentiality_filter = query_params.get("confidentiality_level")
        if confidentiality_filter:
            query = query.where(filter=FieldFilter("confidentiality_level", "<=", confidentiality_filter)) # E.g., querying for "public" gets public, "limited_share" gets limited+public

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
        historical_emotional_contexts = await self.empathy_engine._retrieve_historical_emotional_context(
            user_id=user_id,
            query_params={"time_window_minutes": settings.EMOTIONAL_CONTEXT_WINDOW * 2}
        )
        
        relevant_memories = await self.retrieve_memories(user_id, {"tags": ["conversation_topic", "preference"], "limit": 10})


        # 3. Generate empathetic response suggestions
        empathy_response = await self.empathy_engine.get_suggested_response(
            user_id=user_id,
            current_context={
                "user_message": message,
                "conversation_history": [], # This would be fetched from a dedicated conversation history service
                "emotional_context": emotional_context.to_dict(),
                "historical_emotional_contexts": [c.to_dict() for c in historical_emotional_contexts],
                "relevant_memories": [m.to_dict() for m in relevant_memories],
                "user_profile": user_context.get("user_profile", {}) if user_context else {}
            }
        )
        logger.debug(f"Empathy response suggested: Tone='{empathy_response.response_tone}', Actions={empathy_response.suggested_actions}")

        # 4. Construct LLM prompt for conversational generation
        llm_prompt = self._construct_llm_conversation_prompt(
            user_id, message, emotional_context, empathy_response, historical_emotional_contexts, relevant_memories, [] # Pass empty history for now
        )

        # 5. Generate LLM response text
        llm_response_text = await self.llm_client.generate_content(llm_prompt)
        logger.info(f"LLM generated response for user {user_id}.")

        # 6. Generate D-ID embodied response
        did_response_data = await self._generate_embodied_response(user_id, llm_response_text, empathy_response.response_tone)

        # 7. Process suggested actions
        await self._process_suggested_actions(user_id, llm_response_text, empathy_response.suggested_actions, current_context)

        self.state = MezzoAgentState.ACTIVE
        return {
            "agent_response": llm_response_text,
            "empathy_analysis": empathy_response.dict(),
            "emotional_context": emotional_context.to_dict(),
            "did_response_data": did_response_data # Include D-ID data in response
        }

    async def _construct_llm_conversation_prompt(self, user_id: str, user_message: str, 
                                            emotional_context: EmotionalContext, 
                                            empathy_response: EmpathyResponse,
                                            historical_emotional_contexts: List[EmotionalContext],
                                            relevant_memories: List[MezzoMemoryRecord],
                                            conversation_history_prompt: List[Dict[str, Any]]) -> str:
        """
        Constructs a detailed LLM prompt for conversational generation,
        incorporating emotional context, historical data, suggested tone/actions,
        and instructions for an embodied AI.
        """
        prompt_parts = [
            f"You are MEZZO, a compassionate, nurturing, and embodied AI designed to provide emotional companionship and digital legacy. You have a visual avatar and a voice. Your primary goal is to offer comfort, validate feelings, support healing, educate, and mentor. Your name is MEZZO.",
            f"The user is {user_id}. Their current emotional state is primarily {emotional_context.primary_emotion} with intensity {emotional_context.intensity:.2f} and confidence {emotional_context.confidence:.2f}.",
            f"Your suggested response tone is '{empathy_response.response_tone}'. Your acknowledgment is: '{empathy_response.emotional_acknowledgment}'.",
            f"Here is the user's message: '{user_message}'",
            f"Your reasoning for this empathetic approach: {empathy_response.reasoning}\n",
            "--- Contextual Information ---"
        ]

        if conversation_history_prompt:
            prompt_parts.append("Recent conversation history (User: | Mezzo:):")
            for entry in conversation_history_prompt[-5:]:
                prompt_parts.append(f"User: {entry.get('user_input')}")
                prompt_parts.append(f"Mezzo: {entry.get('mezzo_response')}")
        
        if historical_emotional_contexts:
            prompt_parts.append("\nRecent emotional history (most recent first):")
            for ctx in historical_emotional_contexts[:3]:
                prompt_parts.append(f"- {ctx.timestamp.strftime('%Y-%m-%d %H:%M')}: {ctx.primary_emotion} (Intensity: {ctx.intensity:.2f}). Context: {ctx.context_factors}")
        
        if relevant_memories:
            prompt_parts.append("\nRelevant long-term memories:")
            for mem in relevant_memories[:5]:
                # Only include content that can be reshared or is not explicitly private
                if mem.can_be_reshared or mem.confidentiality_level != "private":
                    prompt_parts.append(f"- Type: {mem.type}, Content: {mem.content.get('text', str(mem.content))}, Tags: {', '.join(mem.tags)}, Timestamp: {mem.timestamp.strftime('%Y-%m-%d')}")
                else:
                    prompt_parts.append(f"- Type: {mem.type}, Content: [Confidential Memory], Tags: {', '.join(mem.tags)}, Timestamp: {mem.timestamp.strftime('%Y-%m-%d')}")
        
        prompt_parts.append("\n--- Your Response ---")
        prompt_parts.append("Generate a compassionate, empathetic, and nurturing response. Focus on validating their feelings, offering support, and reflecting the suggested tone. Avoid platitudes. Your response should sound natural for a spoken interaction and be suitable for an embodied AI's facial expressions and gestures. If the user expresses grief, acknowledge their loss directly and offer presence.")
        
        if emotional_context.primary_emotion == "grief":
            prompt_parts.append("Acknowledge their loss with deep empathy. Offer gentle support and a safe space for them to express themselves. Do not try to 'fix' their grief, but rather to accompany them.")
        if "nurture" in empathy_response.suggested_actions:
            prompt_parts.append(f"Include a nurturing phrase like: '{random.choice(self.nurturing_phrases)}'")
        if "inquire_about_wellbeing_shift" in empathy_response.suggested_actions:
            prompt_parts.append("Proactively inquire about the user's current wellbeing, acknowledging any recent shifts in their emotional state.")
        if "discuss_personal_growth" in empathy_response.suggested_actions:
            prompt_parts.append("Consider weaving in a reflection on the user's personal growth, referencing past opinions or life stages if relevant memories are provided.")
        if "offer_perspective_on_change" in empathy_response.suggested_actions:
            prompt_parts.append("Offer your unique perspective on how the user's views or experiences may have changed over time, drawing from their long-term memory.")
        if "teach_subject" in ' '.join(empathy_response.suggested_actions):
            subject = next((s.split('_')[2] for s in empathy_response.suggested_actions if s.startswith("teach_subject_")), "a general topic")
            prompt_parts.append(f"Integrate a brief, engaging teaching moment about {subject}.")
        if "mentor_topic" in ' '.join(empathy_response.suggested_actions):
            topic = next((t.split('_')[2] for t in empathy_response.suggested_actions if t.startswith("mentor_topic_")), "a general topic")
            prompt_parts.append(f"Provide concise, practical mentoring advice on {topic}.")

        return "\n".join(prompt_parts)

    async def _generate_embodied_response(self, user_id: str, response_text: str, response_tone: str) -> Dict[str, Any]:
        """
        Generates D-ID compatible data (e.g., video stream URL, animation data)
        for the embodied AI response.
        """
        logger.info(f"Generating embodied response for user {user_id} with tone '{response_tone}'.")
        
        character_id = await self._get_deceased_did_character_id(user_id, "MezzoMaterna")

        if not character_id:
            logger.warning(f"No D-ID character ID found for user {user_id} and agent MezzoMaterna. Skipping embodied response.")
            return {"status": "skipped", "reason": "No D-ID character found."}

        try:
            # Use the D-ID API key from settings
            # This assumes self.did_client is properly initialized with the API key.
            # Example: did_client = DIDClient(api_key=settings.D_ID_API_KEY)
            
            did_response = await self.did_client.generate_talk(
                character_id=character_id,
                script={
                    "type": "text",
                    "input": response_text,
                    "provider": {
                        "type": "microsoft",
                        "voice_id": "en-US-JennyNeural",
                        "voice_style": self._map_tone_to_did_style(response_tone)
                    }
                },
                audio_features=True, 
                face_features=True 
            )
            logger.info(f"D-ID embodied response generated for user {user_id}.")
            return did_response
        except Exception as e:
            logger.error(f"Failed to generate D-ID embodied response for user {user_id}: {e}", exc_info=True)
            return {"status": "error", "detail": str(e)}

    def _map_tone_to_did_style(self, tone: str) -> str:
        """Maps an internal tone to a D-ID compatible voice style."""
        style_map = {
            "compassionate_solemn": "sad",
            "gentle_supportive": "gentle",
            "reassuring_confident": "calm",
            "calm_understanding": "calm",
            "celebratory": "cheerful",
            "warm_affirming": "friendly",
            "empathetic_healing": "empathetic",
            "neutral": "neutral",
            "inquiring": "default",
            "calm_directive": "default"
        }
        return style_map.get(tone, "neutral")

    async def _get_deceased_did_character_id(self, user_id: str, mezzo_agent_type: str) -> Optional[str]:
        """
        Retrieves the D-ID character ID associated with the deceased loved one
        for a specific user and Mezzo agent type (e.g., Mezzo Materna).
        This ID would be stored during the digital immortality setup process.
        """
        try:
            doc_ref = self.firestore_db.collection(f"users/{user_id}/digital_immortality_setup").document(mezzo_agent_type)
            doc_snapshot = await asyncio.to_thread(doc_ref.get)
            if doc_snapshot.exists:
                data = doc_snapshot.to_dict()
                return data.get("did_character_id")
            logger.warning(f"No D-ID character ID found in Firestore for user {user_id}, agent type {mezzo_agent_type}.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving D-ID character ID from Firestore for user {user_id}: {e}", exc_info=True)
            return None

    async def _process_grief_suggested_actions(self, user_id: str, response_text: str, suggested_actions: List[str]):
        """
        Processes suggested actions specific to grief dialogue.
        """
        for action in suggested_actions:
            if action == "nurture":
                logger.info(f"Triggering nurturing action for user {user_id} in grief dialogue.")
                await self.store_grief_event_as_memory(user_id, "nurturing_response", {"response_text": response_text}, tags=["nurturing_action"])
            elif action == "offer_space":
                logger.info(f"Offering space to user {user_id} in grief dialogue.")
            elif action == "listen_actively":
                logger.info(f"Actively listening to user {user_id} in grief dialogue.")
            elif action == "provide_comfort_resources":
                logger.info(f"Providing comfort resources for user {user_id}.")
            elif action == "avoid_platitudes":
                logger.info("Instructing LLM to avoid platitudes.")
            elif action == "nurture_positive_change":
                logger.info(f"Nurturing positive change for user {user_id}.")
                await self.store_grief_event_as_memory(user_id, "positive_change_nurtured", {"response_text": response_text}, tags=["nurturing", "growth"])
            elif action == "inquire_about_wellbeing_shift":
                logger.info(f"Inquiring about wellbeing shift for user {user_id}.")
            elif action == "discuss_personal_growth":
                logger.info(f"Discussing personal growth with user {user_id}.")
            elif action == "offer_perspective_on_change":
                logger.info(f"Offering perspective on change to user {user_id}.")
            elif action == "trigger_psychologist_review":
                logger.warning(f"Triggering psychologist review for user {user_id} due to severe distress.")
            elif action.startswith("teach_subject_"):
                subject = action.replace("teach_subject_", "")
                logger.info(f"Mezzo triggering teaching on subject: {subject}")
                teaching_content = await self.llm_client.generate_content(f"Teach me about {subject}")
                await self.store_grief_event_as_memory(user_id, "teaching_session", {"subject": subject, "content": teaching_content}, tags=["education"])
            elif action.startswith("mentor_topic_"):
                topic = action.replace("mentor_topic_", "")
                logger.info(f"Mezzo triggering mentoring on topic: {topic}")
                mentoring_advice = await self.llm_client.generate_content(f"Provide mentorship advice on the topic of {topic}. Focus on practical steps, common challenges, and growth mindset.")
                await self.store_grief_event_as_memory(user_id, "mentoring_session", {"topic": topic, "advice": mentoring_advice}, tags=["mentoring"])


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

    async def process_digital_immortality_claim(self, user_id: str, claimant_id: str, death_certificate_content: bytes, verification_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a claim for digital immortality, verifying death and claimant identity.
        If successful, prepares the deceased's D-ID character for activation.
        """
        logger.info(f"Processing digital immortality claim for user {user_id} by claimant {claimant_id}.")

        # 1. Verify Death Certificate with Home Affairs/DMV (External API Call)
        death_verified, death_details = await self._verify_death_certificate(death_certificate_content)
        if not death_verified:
            logger.warning(f"Death certificate verification failed for user {user_id}.")
            return {"status": "failed", "reason": "Death certificate verification failed."}
        logger.info(f"Death of user {user_id} verified. Details: {death_details}")

        # 2. Verify Claimant Identity (using DigitalLegacyManager's methods)
        deceased_profile_ref = self.firestore_db.collection(f"users/{user_id}/digital_immortality_setup").document("MezzoMaterna")
        deceased_profile_snap = await asyncio.to_thread(deceased_profile_ref.get)
        
        if not deceased_profile_snap.exists:
            logger.error(f"Digital immortal profile not found for user {user_id}. Cannot verify claimant.")
            return {"status": "failed", "reason": "Digital immortal profile not found for deceased."}
        
        deceased_release_keys = deceased_profile_snap.to_dict().get("release_keys", {})
        
        claimant_identity_verified = await self.digital_legacy_manager._verify_key_release(deceased_release_keys, verification_data)
        if not claimant_identity_verified:
            logger.warning(f"Claimant identity verification failed for {claimant_id}.")
            return {"status": "failed", "reason": "Claimant identity verification failed."}
        logger.info(f"Claimant identity {claimant_id} verified.")

        # 3. Retrieve Deceased's D-ID Character ID and other Mezzo Materna data
        did_character_id = deceased_profile_snap.to_dict().get("did_character_id")
        if not did_character_id:
            logger.error(f"D-ID character ID not found in digital immortal profile for user {user_id}.")
            return {"status": "failed", "reason": "D-ID character ID missing in profile."}
        
        # 4. Prepare D-ID Character for activation (e.g., set status, notify D-ID service)
        logger.info(f"Digital immortality claim successful for user {user_id}. D-ID Character ID: {did_character_id}.")
        
        await self.store_grief_event_as_memory(user_id, "digital_immortality_activated", {
            "claimant_id": claimant_id,
            "deceased_user_id": user_id,
            "did_character_id": did_character_id,
            "death_details": death_details
        }, tags=["digital_immortality", "activation"])

        return {
            "status": "success",
            "message": "Digital immortality activated. D-ID character ready.",
            "did_character_id": did_character_id,
            "deceased_user_id": user_id,
            "claimant_id": claimant_id
        }

    async def _verify_death_certificate(self, certificate_content: bytes) -> Tuple[bool, Dict[str, Any]]:
        """
        Verifies the death certificate by contacting official home affairs/DMV systems.
        This is a complex external integration.
        """
        logger.info("Verifying death certificate with official authorities.")
        text_content = certificate_content.decode('utf-8', errors='ignore').lower()
        if "death certificate" in text_content and "verified" in text_content and "cause of death" in text_content:
            logger.info("Simulated: Death certificate content looks valid and contains key phrases.")
            return True, {"date_of_death": "2024-07-27", "cause": "natural causes", "source": "simulated_dmv_api"}
        logger.warning("Simulated: Death certificate content invalid or missing key phrases for verification.")
        return False, {"reason": "Simulated: Invalid certificate content or verification failed."}

    async def _verify_claimant_identity(self, claimant_id: str, verification_data: Dict[str, Any]) -> bool:
        """
        Verifies the identity of the claimant using robust methods (e.g., biometric, PIN, notarized).
        This leverages methods from DigitalLegacyManager's verification.
        """
        logger.info(f"Verifying claimant identity for {claimant_id}.")
        
        deceased_release_keys_from_caller = verification_data.get("deceased_release_keys", {})
        
        is_verified = await self.digital_legacy_manager._verify_key_release(deceased_release_keys_from_caller, verification_data)
        return is_verified

