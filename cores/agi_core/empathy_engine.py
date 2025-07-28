# sans-mercantile-app/backend/agi_core/empathy_engine.py
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import re

from backend.config.settings import settings
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

# Assuming LLMClient is available for teaching/mentoring
from mpeti_modules.ai_ops import LLMClient 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@dataclass
class EmotionalContext:
    primary_emotion: str
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    context_factors: Dict[str, Any]
    timestamp: datetime
    user_id: str = "" 
    event_id: str = ""

    def to_dict(self):
        """Converts the dataclass to a dictionary suitable for Firestore."""
        return {
            "primary_emotion": self.primary_emotion,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "context_factors": self.context_factors,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "event_id": self.event_id
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]):
        """Creates an EmotionalContext from a dictionary (e.g., from Firestore)."""
        return EmotionalContext(
            primary_emotion=data.get("primary_emotion", "neutral"),
            intensity=data.get("intensity", 0.0),
            confidence=data.get("confidence", 0.0),
            context_factors=data.get("context_factors", {}),
            timestamp=data.get("timestamp", datetime.utcnow()),
            user_id=data.get("user_id", ""),
            event_id=data.get("event_id", "")
        )

@dataclass
class EmpathyResponse:
    response_tone: str
    emotional_acknowledgment: str
    suggested_actions: List[str]
    confidence: float
    reasoning: str = ""

class EmotionDetector:
    """Advanced emotion detection from text, context, and multimodal inputs."""
    
    def __init__(self):
        self.emotion_patterns = {
            "joy": [r"\b(happy|joy|excited|thrilled|delighted|pleased|elated)\b", r"😊|😄|😃|🎉|🥳"],
            "sadness": [r"\b(sad|depressed|down|upset|disappointed|hurt|unhappy|miserable)\b", r"😢|😭|☹️|😞|😔"],
            "anger": [r"\b(angry|mad|furious|irritated|annoyed|frustrated|rage)\b", r"😠|😡|🤬|😤"],
            "fear": [r"\b(afraid|scared|worried|anxious|nervous|concerned|terrified)\b", r"😰|😨|😱|😬"],
            "surprise": [r"\b(surprised|shocked|amazed|astonished|unexpected)\b", r"😲|😮|🤯|😳"],
            "disgust": [r"\b(disgusted|revolted|sick|appalled|repulsed)\b", r"🤢|🤮|😷|😒"],
            "trust": [r"\b(trust|confident|secure|safe|reliable|believe)\b", r"🤝|💪|👍"],
            "anticipation": [r"\b(excited|eager|looking forward|anticipating|hopeful)\b", r"🤗|😍|✨"],
            "love": [r"\b(love|adore|cherish|affection|fondness|beloved)\b", r"❤️|🥰|😍"], # New complex emotion
            "grief": [r"\b(grief|mourn|loss|bereaved|heartbroken|devastated)\b", r"💔|🥀"], # New complex emotion
            "heartbreak": [r"\b(heartbreak|crushed|shattered|brokenhearted)\b", r"💔"], # New complex emotion
            "confusion": [r"\b(confused|puzzled|unclear|perplexed)\b", r"🤔|😕"], # New complex emotion
            "empathy": [r"\b(understand|empathize|feel for|relate)\b"], # New complex emotion
            "pride": [r"\b(proud|accomplished|satisfied)\b", r"🏆|🌟"] # New complex emotion
        }
        
        self.intensity_modifiers = {
            "very": 1.3, "extremely": 1.5, "incredibly": 1.4, "really": 1.2,
            "quite": 1.1, "somewhat": 0.8, "slightly": 0.6, "a bit": 0.7,
            "deeply": 1.4, "utterly": 1.5, "barely": 0.5, "profoundly": 1.6 # Added more modifiers
        }
    
    def detect_emotion_from_text(self, text: str, context: Dict[str, Any] = None) -> EmotionalContext:
        """Detect emotion from text with contextual awareness."""
        text_lower = text.lower()
        detected_emotions = {}
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                score += len(matches) * 0.3
            
            if score > 0:
                detected_emotions[emotion] = min(score, 1.0)
        
        for modifier, multiplier in self.intensity_modifiers.items():
            if modifier in text_lower:
                for emotion in detected_emotions:
                    detected_emotions[emotion] *= multiplier
                    detected_emotions[emotion] = min(detected_emotions[emotion], 1.0)
        
        if detected_emotions:
            primary_emotion = max(detected_emotions, key=detected_emotions.get)
            intensity = detected_emotions[primary_emotion]
        else:
            primary_emotion = "neutral"
            intensity = 0.5
        
        confidence = min(intensity * 1.2, 1.0) if detected_emotions else 0.3
        
        context_factors = self._analyze_context(text, context or {})
        
        return EmotionalContext(
            primary_emotion=primary_emotion,
            intensity=intensity,
            confidence=confidence,
            context_factors=context_factors,
            timestamp=datetime.utcnow()
        )
    
    async def detect_emotion_from_audio_analysis(self, audio_features: Dict[str, Any]) -> EmotionalContext:
        """
        Detects emotion from audio analysis features (e.g., tone, pitch, volume).
        This method would integrate with a real audio processing module.
        """
        logger.debug(f"Detecting emotion from audio features: {audio_features}")
        # Enhanced logic for audio-based emotion detection
        if audio_features.get("vocal_stress", 0) > 0.7 and audio_features.get("speech_rate", 0) > 180:
            return EmotionalContext(primary_emotion="Anxious", intensity=0.8, confidence=0.7, context_factors={"audio_cues": "high_stress_fast_speech"}, timestamp=datetime.utcnow())
        if audio_features.get("vocal_pitch", 0) < 0.3 and audio_features.get("volume_db", 0) < 50: # Low pitch, low volume
            return EmotionalContext(primary_emotion="sadness", intensity=0.6, confidence=0.5, context_factors={"audio_cues": "low_pitch_volume"}, timestamp=datetime.utcnow())
        if audio_features.get("laughter_detected", False):
            return EmotionalContext(primary_emotion="joy", intensity=0.9, confidence=0.8, context_factors={"audio_cues": "laughter"}, timestamp=datetime.utcnow())
        return EmotionalContext(primary_emotion="neutral", intensity=0.3, confidence=0.4, context_factors={}, timestamp=datetime.utcnow())

    async def detect_emotion_from_vision_analysis(self, facial_features: Dict[str, Any]) -> EmotionalContext:
        """
        Detects emotion from facial expression analysis features.
        This method would integrate with a real vision processing module.
        """
        logger.debug(f"Detecting emotion from facial features: {facial_features}")
        # Enhanced logic for vision-based emotion detection
        if facial_features.get("brow_furrow", 0) > 0.7 and facial_features.get("mouth_open", 0) < 0.2 and facial_features.get("eye_squint", 0) > 0.5:
            return EmotionalContext(primary_emotion="anger", intensity=0.9, confidence=0.8, context_factors={"facial_cues": "anger_indicators"}, timestamp=datetime.utcnow())
        if facial_features.get("smile_intensity", 0) > 0.8 and facial_features.get("eye_crinkle", 0) > 0.5:
            return EmotionalContext(primary_emotion="joy", intensity=0.9, confidence=0.8, context_factors={"facial_cues": "true_smile"}, timestamp=datetime.utcnow())
        if facial_features.get("inner_brow_raise", 0) > 0.6 and facial_features.get("lip_corner_depressor", 0) > 0.6:
            return EmotionalContext(primary_emotion="sadness", intensity=0.7, confidence=0.6, context_factors={"facial_cues": "sadness_indicators"}, timestamp=datetime.utcnow())
        return EmotionalContext(primary_emotion="neutral", intensity=0.3, confidence=0.4, context_factors={}, timestamp=datetime.utcnow())

    async def detect_emotion_from_system_status(self, status_data: Dict[str, Any]) -> EmotionalContext:
        """Detects emotion from system status."""
        logger.debug(f"Detecting emotion from system status: {status_data}")
        if status_data.get("critical_alerts_count", 0) > 0 or status_data.get("status") == "FAILED":
            return EmotionalContext("Tense", 0.9, 0.8, {"system_alerts": status_data.get("critical_alerts_count")}, datetime.utcnow())
        return EmotionalContext("Calm", 0.2, 0.7, {}, datetime.utcnow())

    async def detect_emotion_from_environmental_data(self, env_data: Dict[str, Any]) -> EmotionalContext:
        """Detects emotion from environmental data (e.g., noise levels, light)."""
        logger.debug(f"Detecting emotion from environmental data: {env_data}")
        if env_data.get("sound_level_db", 0) > 70 or env_data.get("light_lux", 0) < 50:
            return EmotionalContext("Unsettled", 0.6, 0.5, {"environment_cues": "noisy_dark"}, datetime.utcnow())
        return EmotionalContext("neutral", 0.1, 0.9, {}, datetime.utcnow())
    
    def _analyze_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contextual factors that might influence emotion."""
        factors = {}
        
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            factors["time_stress"] = "late_hours"
        elif 9 <= current_hour <= 17:
            factors["time_context"] = "work_hours"
        
        financial_keywords = ["money", "loss", "profit", "investment", "trade", "market"]
        if any(keyword in text.lower() for keyword in financial_keywords):
            factors["domain"] = "financial"
        
        urgency_keywords = ["urgent", "asap", "immediately", "quickly", "emergency"]
        if any(keyword in text.lower() for keyword in urgency_keywords):
            factors["urgency"] = "high"
        
        if context.get("environmental_data"):
            env_data = context["environmental_data"]
            if env_data.get("stress_level", 0) > 0.7:
                factors["environmental_stress"] = "high"
        
        # Add specific context factors for complex emotions
        if "relationship_status" in context:
            factors["relationship_status"] = context["relationship_status"]
        if "life_event" in context:
            factors["life_event"] = context["life_event"] # e.g., "bereavement", "celebration"
        
        return factors

class EmpathyEngine:
    """Advanced empathy engine for emotional intelligence."""
    
    def __init__(self, broker, firestore_db: firestore.Client):
        self.broker = broker
        self.firestore_db = firestore_db
        self.emotion_detector = EmotionDetector()
        self.empathy_strategies = self._load_empathy_strategies()
        self.is_active = settings.EMPATHY_ENGINE_ENABLED
        self.current_overall_mood: Optional[EmotionalContext] = None
        self.llm_client = LLMClient() # For teaching/mentoring via LLM

    def _load_empathy_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load empathy response strategies for different emotions."""
        return {
            "joy": {
                "tone": "celebratory",
                "acknowledgment": "I can sense your excitement and happiness!",
                "actions": ["amplify_positive_sentiment", "share_enthusiasm", "encourage_continuation"]
            },
            "sadness": {
                "tone": "gentle_supportive",
                "acknowledgment": "I understand you're going through a difficult time.",
                "actions": ["provide_comfort", "offer_practical_help", "validate_feelings", "suggest_reflection"]
            },
            "anger": {
                "tone": "calm_understanding",
                "acknowledgment": "I can see that you're feeling frustrated about this situation.",
                "actions": ["de_escalate", "find_solutions", "acknowledge_validity", "suggest_cooling_off"]
            },
            "fear": {
                "tone": "reassuring_confident",
                "acknowledgment": "I recognize your concerns and they're completely valid.",
                "actions": ["provide_reassurance", "break_down_problems", "offer_support", "offer_information"]
            },
            "surprise": {
                "tone": "curious_engaging",
                "acknowledgment": "Oh, that's quite surprising!",
                "actions": ["seek_clarification", "explore_reason"]
            },
            "disgust": {
                "tone": "neutral_concerned",
                "acknowledgment": "I note your strong negative reaction.",
                "actions": ["understand_source", "address_cause"]
            },
            "trust": {
                "tone": "affirming_reliable",
                "acknowledgment": "I appreciate your trust in me.",
                "actions": ["maintain_transparency", "reinforce_reliability", "offer_continued_support"]
            },
            "anticipation": {
                "tone": "enthusiastic_supportive",
                "acknowledgment": "I can feel your anticipation!",
                "actions": ["encourage_progress", "prepare_for_outcome", "offer_guidance"]
            },
            "love": { # Strategy for love
                "tone": "warm_affirming",
                "acknowledgment": "I sense a deep connection and warmth in your words.",
                "actions": ["reciprocate_positive_sentiment", "affirm_connection", "express_care"]
            },
            "grief": { # Strategy for grief
                "tone": "compassionate_solemn",
                "acknowledgment": "I am deeply sorry for your loss. Please know I am here for you.",
                "actions": ["offer_space", "listen_actively", "provide_comfort_resources", "avoid_platitudes", "nurture"]
            },
            "heartbreak": { # Strategy for heartbreak
                "tone": "empathetic_healing",
                "acknowledgment": "I understand this pain must be immense. It takes courage to share this.",
                "actions": ["validate_pain", "offer_distraction", "suggest_support_networks", "encourage_self_care"]
            },
            "confusion": { # Strategy for confusion
                "tone": "clarifying_patient",
                "acknowledgment": "It seems there's some confusion, and I'm here to help clarify.",
                "actions": ["break_down_information", "ask_clarifying_questions", "rephrase_explanation"]
            },
            "pride": { # Strategy for pride
                "tone": "congratulatory_affirming",
                "acknowledgment": "That's truly impressive! You should be very proud.",
                "actions": ["acknowledge_achievement", "encourage_sharing", "reinforce_self_worth"]
            },
            "neutral": {
                "tone": "balanced_helpful",
                "acknowledgment": "I'm here to help you with whatever you need.",
                "actions": ["assess_needs", "provide_information", "maintain_engagement"]
            },
            "Unsettled": {
                "tone": "inquiring",
                "acknowledgment": "I sense a slight unease, perhaps related to the environment.",
                "actions": ["identify_environmental_cause", "offer_comfort"]
            },
            "Tense": {
                "tone": "calm_directive",
                "acknowledgment": "I detect a tense atmosphere. Let's focus on stability.",
                "actions": ["de_escalate", "prioritize_critical_tasks"]
            }
        }
        
    async def analyze_emotional_context(self, user_id: str, text: str = None, audio_features: Dict[str, Any] = None, 
                                        vision_features: Dict[str, Any] = None, system_status: Dict[str, Any] = None, 
                                        environmental_data: Dict[str, Any] = None, user_context: Dict[str, Any] = None) -> EmotionalContext:
        """
        Analyze emotional context from various multimodal inputs.
        Combines inputs to form a holistic emotional context and stores it persistently.
        """
        if not self.is_active:
            return EmotionalContext("neutral", 0.5, 0.3, {}, datetime.utcnow(), user_id=user_id, event_id="")
        
        emotional_contexts: List[EmotionalContext] = []

        if text:
            emotional_contexts.append(self.emotion_detector.detect_emotion_from_text(text, user_context))
        if audio_features:
            emotional_contexts.append(await self.emotion_detector.detect_emotion_from_audio_analysis(audio_features))
        if vision_features:
            emotional_contexts.append(await self.emotion_detector.detect_emotion_from_vision_analysis(vision_features))
        if system_status:
            emotional_contexts.append(await self.emotion_detector.detect_emotion_from_system_status(system_status))
        if environmental_data:
            emotional_contexts.append(await self.emotion_detector.detect_emotion_from_environmental_data(environmental_data))

        holistic_context = self._fuse_emotional_contexts(emotional_contexts)
        holistic_context.user_id = user_id
        holistic_context.event_id = f"emotion_{user_id}_{holistic_context.timestamp.timestamp()}"

        try:
            doc_ref = self.firestore_db.collection(f"users/{user_id}/emotional_history").document(holistic_context.event_id)
            await asyncio.to_thread(doc_ref.set, holistic_context.to_dict())
            logger.info(f"Emotional context saved to Firestore for user {user_id}: {holistic_context.primary_emotion}")
        except Exception as e:
            logger.error(f"Failed to save emotional context to Firestore for user {user_id}: {e}", exc_info=True)
        
        await self._update_overall_mood(user_id)
        
        return holistic_context

    def _fuse_emotional_contexts(self, contexts: List[EmotionalContext]) -> EmotionalContext:
        """
        Fuses multiple emotional contexts from different modalities into a single holistic context.
        """
        if not contexts:
            return EmotionalContext("neutral", 0.0, 0.0, {}, datetime.utcnow())

        fused_mood_scores = {}
        fused_intensity = 0.0
        fused_confidence = 0.0
        fused_context_factors = {}
        total_weight = 0.0

        source_weights = {
            "user_input": 1.0, "audio_cues": 0.8, "facial_cues": 0.8,
            "system_status": 0.6, "environmental_sensor": 0.5
        }

        for ctx in contexts:
            weight = source_weights.get(ctx.context_factors.get("source", "user_input"), 0.5)
            
            fused_intensity += ctx.intensity * weight
            fused_confidence += ctx.confidence * weight
            total_weight += weight

            fused_mood_scores[ctx.primary_emotion] = fused_mood_scores.get(ctx.primary_emotion, 0) + (ctx.intensity * weight)

            fused_context_factors.update(ctx.context_factors)

        if total_weight == 0:
            return EmotionalContext("neutral", 0.0, 0.0, {}, datetime.utcnow())

        fused_intensity /= total_weight
        fused_confidence /= total_weight
        
        dominant_emotion = max(fused_mood_scores, key=fused_mood_scores.get) if fused_mood_scores else "neutral"

        return EmotionalContext(
            primary_emotion=dominant_emotion,
            intensity=fused_intensity,
            confidence=fused_confidence,
            context_factors=fused_context_factors,
            timestamp=datetime.utcnow()
        )

    async def _update_overall_mood(self, user_id: str):
        """
        Aggregates recent emotional contexts from Firestore to determine an overall system mood.
        """
        recent_history = await self._retrieve_historical_emotional_context(
            user_id=user_id,
            query_params={"time_window_minutes": settings.EMOTIONAL_CONTEXT_WINDOW}
        )

        if not recent_history:
            self.current_overall_mood = EmotionalContext("neutral", 0.0, 0.0, {}, datetime.utcnow(), user_id=user_id)
            return

        mood_scores = {
            "joy": 0, "sadness": 0, "anger": 0, "fear": 0, "surprise": 0,
            "disgust": 0, "trust": 0, "anticipation": 0, "neutral": 0,
            "Unsettled": 0, "Tense": 0, "Calm": 0, "love": 0, "grief": 0, "heartbreak": 0, "confusion": 0, "empathy": 0, "pride": 0
        }
        total_intensity = 0.0
        total_samples = 0

        for context in recent_history:
            score = context.intensity * context.confidence
            mood_scores[context.primary_emotion] = mood_scores.get(context.primary_emotion, 0) + score
            total_intensity += context.intensity
            total_samples += 1

        dominant_mood = max(mood_scores, key=mood_scores.get) if mood_scores else "neutral"
        avg_intensity = total_intensity / total_samples if total_samples > 0 else 0.0
        avg_confidence = sum(ctx.confidence for ctx in recent_history) / total_samples if total_samples > 0 else 0.0
        
        self.current_overall_mood = EmotionalContext(
            primary_emotion=dominant_mood,
            intensity=avg_intensity,
            confidence=avg_confidence,
            context_factors={"aggregation_method": "weighted_average_from_firestore"},
            timestamp=datetime.utcnow(),
            user_id=user_id
        )
        logger.debug(f"Overall mood updated for user {user_id}: {self.current_overall_mood.primary_emotion} (Intensity: {self.current_overall_mood.intensity:.2f}, Confidence: {self.current_overall_mood.confidence:.2f})")

    async def _retrieve_historical_emotional_context(self, user_id: str, query_params: Dict[str, Any]) -> List[EmotionalContext]:
        """
        Retrieves historical EmotionalContext records from Firestore for long-term memory.
        Query parameters can include 'start_date', 'end_date', 'emotions', 'keywords', 'time_window_minutes', 'event_type'.
        """
        collection_ref = self.firestore_db.collection(f"users/{user_id}/emotional_history")
        query = collection_ref.order_by("timestamp", direction=firestore.Query.DESCENDING)
        
        time_window_minutes = query_params.get("time_window_minutes")
        if time_window_minutes:
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            query = query.where(filter=FieldFilter("timestamp", ">=", cutoff_time))
            logger.debug(f"Retrieving emotional history for user {user_id} within last {time_window_minutes} minutes.")
        
        start_date = query_params.get("start_date")
        end_date = query_params.get("end_date")
        if start_date:
            query = query.where(filter=FieldFilter("timestamp", ">=", start_date))
        if end_date:
            query = query.where(filter=FieldFilter("timestamp", "<=", end_date))

        emotions_filter = query_params.get("emotions")
        if emotions_filter and isinstance(emotions_filter, list):
            query = query.where(filter=FieldFilter("primary_emotion", "in", emotions_filter))

        event_type_filter = query_params.get("event_type")
        if event_type_filter:
            query = query.where(filter=FieldFilter("context_factors.event_type", "==", event_type_filter))

        limit = query_params.get("limit", 100)
        query = query.limit(limit)

        try:
            docs = await asyncio.to_thread(query.stream)
            history = [EmotionalContext.from_dict(doc.to_dict()) for doc in docs]
            logger.info(f"Retrieved {len(history)} historical emotional contexts for user {user_id}.")
            return history
        except Exception as e:
            logger.error(f"Failed to retrieve historical emotional context for user {user_id}: {e}", exc_info=True)
            return []

    async def get_suggested_response(self, user_id: str, current_context: Dict[str, Any], ethical_evaluation: Optional[Dict[str, Any]] = None) -> EmpathyResponse:
        """
        Suggests an empathetic response based on the current overall mood, context,
        ethical evaluation, and long-term emotional history.
        """
        logger.info(f"EmpathyEngine: Generating suggested response for user {user_id} with context: {current_context}")
        
        await self._update_overall_mood(user_id)
        overall_mood = self.current_overall_mood or EmotionalContext("neutral", 0.0, 0.0, {}, datetime.utcnow(), user_id=user_id)

        response_tone = "balanced_helpful"
        emotional_acknowledgment = self.empathy_strategies["neutral"]["acknowledgment"]
        suggested_actions = self.empathy_strategies["neutral"]["actions"].copy()
        reasoning = f"Based on current overall mood: {overall_mood.primary_emotion} (Intensity: {overall_mood.intensity:.2f}, Confidence: {overall_mood.confidence:.2f})."

        strategy = self.empathy_strategies.get(overall_mood.primary_emotion, self.empathy_strategies["neutral"])
        response_tone = strategy["tone"]
        emotional_acknowledgment = strategy["acknowledgment"]
        suggested_actions = strategy["actions"].copy()

        intensity_factor = overall_mood.intensity
        if intensity_factor > 0.8:
            suggested_actions.append("provide_immediate_support")
            reasoning += " High emotional intensity detected."
        
        if "financial" in overall_mood.context_factors.get("domain", ""):
            emotional_acknowledgment += " I understand this involves your financial well-being."
            suggested_actions.append("offer_financial_guidance")
            reasoning += " Financial domain context."
        
        if overall_mood.context_factors.get("urgency") == "high":
            emotional_acknowledgment += " I can see this is time-sensitive for you."
            suggested_actions.append("prioritize_response_speed")
            reasoning += " High urgency context."
        
        if "environmental_stress" in overall_mood.context_factors:
            emotional_acknowledgment += " I notice you might be dealing with additional stress right now."
            suggested_actions.append("offer_environmental_comfort")
            reasoning += " Environmental stress detected."

        if ethical_evaluation and not ethical_evaluation.get("is_compliant", True):
            response_tone = "cautionary"
            suggested_actions.insert(0, "reconsider_action_due_to_ethical_concerns")
            reasoning += f" Ethical concerns raised: {ethical_evaluation.get('violation_details')}. Reconsidering action."
            emotional_acknowledgment = "I must highlight some ethical considerations here."

        # --- Leverage Long-Term Memory for Deeper Personalization ---
        # Example 1: Recall past emotional states for a specific period
        past_week_moods = await self._retrieve_historical_emotional_context(
            user_id=user_id,
            query_params={"time_window_minutes": 7 * 24 * 60} # Last 7 days
        )
        if past_week_moods:
            past_emotions_set = {ctx.primary_emotion for ctx in past_week_moods}
            if "sadness" in past_emotions_set or "fear" in past_emotions_set:
                emotional_acknowledgment += " I remember you faced some challenges recently."
                reasoning += " Recalled past week's negative emotions."
                suggested_actions.append("check_on_recent_wellbeing")
            if "joy" in past_emotions_set and overall_mood.primary_emotion == "neutral":
                 emotional_acknowledgment += " It's good to see you're doing well, especially after a positive week."
                 reasoning += "Acknowledged recent positive trend."

        # Example 2: Recall specific events or long-term preferences/changes (e.g., "When you were 8 years old...")
        # This requires storing events with specific metadata (e.g., age, topic, sentiment)
        # and then querying for them.
        
        # Scenario: User was feeling unwell recently, but now seems better.
        # Query for recent negative emotions (e.g., in the last month)
        recent_negative_history = await self._retrieve_historical_emotional_context(
            user_id=user_id,
            query_params={"time_window_minutes": 30 * 24 * 60, "emotions": ["sadness", "anger", "fear", "grief", "heartbreak"]}
        )
        if recent_negative_history and overall_mood.primary_emotion in ["joy", "neutral", "calm"]:
            # If there's recent negative history but current mood is positive/neutral
            emotional_acknowledgment += " I noticed that you weren't feeling well recently, but you seem much better now. How do you feel today?"
            reasoning += " Detected shift from recent negative emotions to current positive/neutral state."
            suggested_actions.append("inquire_about_wellbeing_shift")
            suggested_actions.append("nurture_positive_change")

        # Scenario: Change in preference over time (e.g., dark humor)
        # This requires specific event logging about preferences/opinions
        # Example: Query for past opinions on "dark humor"
        past_dark_humor_opinions = await self._retrieve_historical_emotional_context(
            user_id=user_id,
            query_params={"event_type": "opinion_expressed", "keywords": ["dark humor"]},
            limit=50 # Fetch enough to find older records
        )
        if past_dark_humor_opinions:
            # Sort to find oldest and newest relevant opinions
            past_dark_humor_opinions.sort(key=lambda x: x.timestamp)
            oldest_opinion = next((op for op in past_dark_humor_opinions if "offensive" in op.context_factors.get("opinion_text", "")), None)
            newest_opinion = next((op for op in reversed(past_dark_humor_opinions) if "like" in op.context_factors.get("opinion_text", "")), None)

            if oldest_opinion and newest_opinion and oldest_opinion.timestamp < newest_opinion.timestamp - timedelta(days=365*5): # At least 5 years difference
                # Rough age calculation if user's birthdate is known and stored in user_context
                user_birth_year = current_context.get("user_profile", {}).get("birth_year")
                if user_birth_year:
                    old_age_approx = oldest_opinion.timestamp.year - user_birth_year
                    if 7 <= old_age_approx <= 9: # Roughly 8 years old
                        emotional_acknowledgment += f" When you were around {old_age_approx} years old, I recall you found dark humor offensive, but now you seem to like it. What's changed in your perspective?"
                        reasoning += " Identified long-term preference change on dark humor based on age approximation."
                        suggested_actions.append("discuss_personal_growth")
                        suggested_actions.append("offer_perspective_on_change")


        # --- Nurturing and Grief Understanding ---
        if overall_mood.primary_emotion == "grief":
            emotional_acknowledgment = self.empathy_strategies["grief"]["acknowledgment"]
            suggested_actions.extend(self.empathy_strategies["grief"]["actions"])
            reasoning += " Detected grief, activating compassionate and nurturing response."
            suggested_actions.append("nurture_emotional_wellbeing") # Explicit nurturing action
            suggested_actions.append("offer_space_and_time")

        # --- Teaching and Mentoring ---
        if current_context.get("user_query_type") == "learning_request":
            subject = current_context.get("subject")
            topic = current_context.get("topic")
            if subject and topic:
                suggested_actions.append(f"teach_subject_{subject}")
                suggested_actions.append(f"mentor_topic_{topic}")
                reasoning += f" User requested learning on {subject} and mentoring on {topic}."
                emotional_acknowledgment += " I'm ready to help you learn and grow."
        
        response_confidence = min(
            overall_mood.confidence * (1.0 + overall_mood.intensity * 0.2) * (1.0 if len(past_week_moods) > 5 else 0.8),
            1.0
        )
        
        return EmpathyResponse(
            response_tone=response_tone,
            emotional_acknowledgment=emotional_acknowledgment,
            suggested_actions=suggested_actions[:8], # Increased limit for more actions
            confidence=response_confidence,
            reasoning=reasoning
        )
        
    def _analyze_emotional_trend(self) -> List[str]:
        """Analyze recent emotional trends from history."""
        return []

    def _generate_contextual_acknowledgment(
        self, 
        emotional_context: EmotionalContext, 
        base_acknowledgment: str,
        recent_emotions: List[str]
    ) -> str:
        """Generate contextual emotional acknowledgment."""
        acknowledgment = base_acknowledgment
        
        if "financial" in emotional_context.context_factors.get("domain", ""):
            acknowledgment += " I understand this involves your financial well-being."
        
        if emotional_context.context_factors.get("urgency") == "high":
            acknowledgment += " I can see this is time-sensitive for you."
        
        if "environmental_stress" in emotional_context.context_factors:
            acknowledgment += " I notice you might be dealing with additional stress right now."
        
        if len(recent_emotions) > 1:
            if recent_emotions[-2] == "anger" and emotional_context.primary_emotion == "sadness":
                acknowledgment += " I can see your feelings have shifted, and that's completely natural."
        
        return acknowledgment
    
    def _select_appropriate_actions(
        self,
        emotional_context: EmotionalContext,
        base_actions: List[str],
        recent_emotions: List[str]
    ) -> List[str]:
        """Select appropriate empathic actions."""
        actions = base_actions.copy()
        
        if emotional_context.intensity > 0.8:
            actions.append("provide_immediate_support")
        
        if "financial" in emotional_context.context_factors.get("domain", ""):
            actions.append("offer_financial_guidance")
        
        if emotional_context.context_factors.get("urgency") == "high":
            actions.append("prioritize_response_speed")
        
        if len(recent_emotions) > 2 and all(e in ["sadness", "fear", "anger"] for e in recent_emotions[-3:]):
            actions.append("suggest_break_or_support")
        
        return actions[:5]
    
    async def adapt_communication_style(self, empathy_response: EmpathyResponse) -> Dict[str, Any]:
        """Adapt communication style based on empathic analysis."""
        adaptations = {
            "tone": empathy_response.response_tone,
            "pace": "normal",
            "detail_level": "standard",
            "supportiveness": "standard"
        }
        
        if empathy_response.response_tone == "gentle_supportive":
            adaptations.update({
                "pace": "slower",
                "detail_level": "simplified",
                "supportiveness": "high"
            })
        elif empathy_response.response_tone == "celebratory":
            adaptations.update({
                "pace": "energetic",
                "detail_level": "enthusiastic",
                "supportiveness": "encouraging"
            })
        elif empathy_response.response_tone == "calm_understanding":
            adaptations.update({
                "pace": "measured",
                "detail_level": "clear_structured",
                "supportiveness": "validating"
            })
        
        return adaptations
    
    def get_emotional_insights(self) -> Dict[str, Any]:
        """Get insights from emotional interaction history."""
        if not self.current_overall_mood:
            return {"status": "no_data"}
        
        return {
            "status": "data_from_current_overall_mood",
            "dominant_emotion": self.current_overall_mood.primary_emotion,
            "average_intensity": round(self.current_overall_mood.intensity, 2),
            "average_confidence": round(self.current_overall_mood.confidence, 2),
            "last_update": self.current_overall_mood.timestamp.isoformat()
        }

