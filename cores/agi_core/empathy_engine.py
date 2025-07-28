# sans-mercantile-app/backend/agi_core/empathy_engine.py
import asyncio # Added for async operations
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import re

from backend.config.settings import settings

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

@dataclass
class EmpathyResponse:
    response_tone: str
    emotional_acknowledgment: str
    suggested_actions: List[str]
    confidence: float
    reasoning: str = "" # Added reasoning field

class EmotionDetector:
    """Advanced emotion detection from text, context, and multimodal inputs."""
    
    def __init__(self):
        self.emotion_patterns = {
            "joy": [r"\b(happy|joy|excited|thrilled|delighted|pleased)\b", r"😊|😄|😃|🎉"],
            "sadness": [r"\b(sad|depressed|down|upset|disappointed|hurt)\b", r"😢|😭|☹️|😞"],
            "anger": [r"\b(angry|mad|furious|irritated|annoyed|frustrated)\b", r"😠|😡|🤬"],
            "fear": [r"\b(afraid|scared|worried|anxious|nervous|concerned)\b", r"😰|😨|😱"],
            "surprise": [r"\b(surprised|shocked|amazed|astonished)\b", r"😲|😮|🤯"],
            "disgust": [r"\b(disgusted|revolted|sick|appalled)\b", r"🤢|🤮|😷"],
            "trust": [r"\b(trust|confident|secure|safe|reliable)\b", r"🤝|💪"],
            "anticipation": [r"\b(excited|eager|looking forward|anticipating)\b", r"🤗|😍"]
        }
        
        self.intensity_modifiers = {
            "very": 1.3, "extremely": 1.5, "incredibly": 1.4, "really": 1.2,
            "quite": 1.1, "somewhat": 0.8, "slightly": 0.6, "a bit": 0.7
        }
    
    def detect_emotion_from_text(self, text: str, context: Dict[str, Any] = None) -> EmotionalContext:
        """Detect emotion from text with contextual awareness."""
        text_lower = text.lower()
        detected_emotions = {}
        
        # Pattern-based detection
        for emotion, patterns in self.emotion_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                score += len(matches) * 0.3
            
            if score > 0:
                detected_emotions[emotion] = min(score, 1.0)
        
        # Apply intensity modifiers
        for modifier, multiplier in self.intensity_modifiers.items():
            if modifier in text_lower:
                for emotion in detected_emotions:
                    detected_emotions[emotion] *= multiplier
                    detected_emotions[emotion] = min(detected_emotions[emotion], 1.0)
        
        # Determine primary emotion
        if detected_emotions:
            primary_emotion = max(detected_emotions, key=detected_emotions.get)
            intensity = detected_emotions[primary_emotion]
        else:
            primary_emotion = "neutral"
            intensity = 0.5
        
        # Calculate confidence based on clarity of emotional signals
        confidence = min(intensity * 1.2, 1.0) if detected_emotions else 0.3
        
        # Consider contextual factors
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
        # Placeholder for actual audio-based emotion detection logic
        if audio_features.get("pitch_variance", 0) > 0.5 and audio_features.get("volume_db", 0) > 70:
            return EmotionalContext(primary_emotion="Anxious", intensity=0.7, confidence=0.6, context_factors={"audio_cues": "high_pitch_volume"}, timestamp=datetime.utcnow())
        if audio_features.get("speech_rate", 0) > 180: # words per minute
            return EmotionalContext(primary_emotion="Tense", intensity=0.6, confidence=0.5, context_factors={"audio_cues": "fast_speech"}, timestamp=datetime.utcnow())
        return EmotionalContext(primary_emotion="neutral", intensity=0.3, confidence=0.4, context_factors={}, timestamp=datetime.utcnow())

    async def detect_emotion_from_vision_analysis(self, facial_features: Dict[str, Any]) -> EmotionalContext:
        """
        Detects emotion from facial expression analysis features.
        This method would integrate with a real vision processing module.
        """
        logger.debug(f"Detecting emotion from facial features: {facial_features}")
        # Placeholder for actual vision-based emotion detection logic
        if facial_features.get("brow_furrow", 0) > 0.7 and facial_features.get("mouth_open", 0) < 0.2:
            return EmotionalContext(primary_emotion="anger", intensity=0.8, confidence=0.7, context_factors={"facial_cues": "furrowed_brow"}, timestamp=datetime.utcnow())
        if facial_features.get("smile_intensity", 0) > 0.8:
            return EmotionalContext(primary_emotion="joy", intensity=0.9, confidence=0.8, context_factors={"facial_cues": "strong_smile"}, timestamp=datetime.utcnow())
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
        
        # Time-based context
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            factors["time_stress"] = "late_hours"
        elif 9 <= current_hour <= 17:
            factors["time_context"] = "work_hours"
        
        # Financial context
        financial_keywords = ["money", "loss", "profit", "investment", "trade", "market"]
        if any(keyword in text.lower() for keyword in financial_keywords):
            factors["domain"] = "financial"
        
        # Urgency indicators
        urgency_keywords = ["urgent", "asap", "immediately", "quickly", "emergency"]
        if any(keyword in text.lower() for keyword in urgency_keywords):
            factors["urgency"] = "high"
        
        # Environmental context from settings
        if context.get("environmental_data"):
            env_data = context["environmental_data"]
            if env_data.get("stress_level", 0) > 0.7:
                factors["environmental_stress"] = "high"
        
        return factors

class EmpathyEngine:
    """Advanced empathy engine for emotional intelligence."""
    
    def __init__(self, broker): # Broker is likely a MessageBrokerInterface instance
        self.broker = broker
        self.emotion_detector = EmotionDetector()
        self.emotional_history: List[EmotionalContext] = []
        self.empathy_strategies = self._load_empathy_strategies()
        self.is_active = settings.EMPATHY_ENGINE_ENABLED
        self.current_overall_mood: Optional[EmotionalContext] = None # Added for overall mood tracking
        
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
                "actions": ["provide_comfort", "offer_practical_help", "validate_feelings"]
            },
            "anger": {
                "tone": "calm_understanding",
                "acknowledgment": "I can see that you're feeling frustrated about this situation.",
                "actions": ["de_escalate", "find_solutions", "acknowledge_validity"]
            },
            "fear": {
                "tone": "reassuring_confident",
                "acknowledgment": "I recognize your concerns and they're completely valid.",
                "actions": ["provide_reassurance", "break_down_problems", "offer_support"]
            },
            "surprise": { # Added surprise
                "tone": "curious_engaging",
                "acknowledgment": "Oh, that's quite surprising!",
                "actions": ["seek_clarification", "explore_reason"]
            },
            "disgust": { # Added disgust
                "tone": "neutral_concerned",
                "acknowledgment": "I note your strong negative reaction.",
                "actions": ["understand_source", "address_cause"]
            },
            "trust": { # Added trust
                "tone": "affirming_reliable",
                "acknowledgment": "I appreciate your trust in me.",
                "actions": ["maintain_transparency", "reinforce_reliability"]
            },
            "anticipation": { # Added anticipation
                "tone": "enthusiastic_supportive",
                "acknowledgment": "I can feel your anticipation!",
                "actions": ["encourage_progress", "prepare_for_outcome"]
            },
            "neutral": {
                "tone": "balanced_helpful",
                "acknowledgment": "I'm here to help you with whatever you need.",
                "actions": ["assess_needs", "provide_information", "maintain_engagement"]
            },
            "Unsettled": { # Added Unsettled mood from environmental data
                "tone": "inquiring",
                "acknowledgment": "I sense a slight unease, perhaps related to the environment.",
                "actions": ["identify_environmental_cause", "offer_comfort"]
            },
            "Tense": { # Added Tense mood from system status
                "tone": "calm_directive",
                "acknowledgment": "I detect a tense atmosphere. Let's focus on stability.",
                "actions": ["de_escalate", "prioritize_critical_tasks"]
            }
        }
        
    async def analyze_emotional_context(self, text: str = None, audio_features: Dict[str, Any] = None, 
                                        vision_features: Dict[str, Any] = None, system_status: Dict[str, Any] = None, 
                                        environmental_data: Dict[str, Any] = None, user_context: Dict[str, Any] = None) -> EmotionalContext:
        """
        Analyze emotional context from various multimodal inputs.
        Combines inputs to form a holistic emotional context.
        """
        if not self.is_active:
            return EmotionalContext("neutral", 0.5, 0.3, {}, datetime.utcnow())
        
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

        # Combine multiple emotional contexts into a single holistic context
        holistic_context = self._fuse_emotional_contexts(emotional_contexts)
        
        self.emotional_history.append(holistic_context)
        
        # Keep only recent history
        cutoff_time = datetime.utcnow() - timedelta(minutes=settings.EMOTIONAL_CONTEXT_WINDOW)
        self.emotional_history = [
            ctx for ctx in self.emotional_history 
            if ctx.timestamp > cutoff_time
        ]
        
        await self._update_overall_mood() # Update overall mood after processing new input
        
        return holistic_context

    def _fuse_emotional_contexts(self, contexts: List[EmotionalContext]) -> EmotionalContext:
        """
        Fuses multiple emotional contexts from different modalities into a single holistic context.
        This is where 'nuanced interpretation' and 'fusion' would happen.
        """
        if not contexts:
            return EmotionalContext("neutral", 0.0, 0.0, {}, datetime.utcnow())

        # Simple fusion: weighted average of intensity and confidence
        # More advanced fusion would use ML models to combine signals
        fused_mood_scores = {}
        fused_intensity = 0.0
        fused_confidence = 0.0
        fused_context_factors = {}
        total_weight = 0.0

        # Assign weights based on source reliability/importance (conceptual)
        source_weights = {
            "user_input": 1.0, "audio_cues": 0.8, "facial_cues": 0.8,
            "system_status": 0.6, "environmental_sensor": 0.5
        }

        for ctx in contexts:
            weight = source_weights.get(ctx.context_factors.get("source", "user_input"), 0.5) # Default weight
            
            fused_intensity += ctx.intensity * weight
            fused_confidence += ctx.confidence * weight
            total_weight += weight

            fused_mood_scores[ctx.primary_emotion] = fused_mood_scores.get(ctx.primary_emotion, 0) + (ctx.intensity * weight)

            # Merge context factors, prioritizing more specific/recent ones
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

    async def _update_overall_mood(self):
        """
        Aggregates recent emotional contexts from history to determine an overall system mood.
        This is where 'nuanced interpretation' and 'fusion' would happen.
        """
        if not self.emotional_history:
            self.current_overall_mood = EmotionalContext("neutral", 0.0, 0.0, {}, datetime.utcnow())
            return

        mood_scores = {
            "joy": 0, "sadness": 0, "anger": 0, "fear": 0, "surprise": 0,
            "disgust": 0, "trust": 0, "anticipation": 0, "neutral": 0,
            "Unsettled": 0, "Tense": 0, "Calm": 0 # Include all possible moods
        }
        total_intensity = 0.0
        total_samples = 0

        for context in self.emotional_history:
            score = context.intensity * context.confidence # Weight by confidence
            mood_scores[context.primary_emotion] = mood_scores.get(context.primary_emotion, 0) + score
            total_intensity += context.intensity
            total_samples += 1

        dominant_mood = max(mood_scores, key=mood_scores.get) if mood_scores else "neutral"
        avg_intensity = total_intensity / total_samples if total_samples > 0 else 0.0
        avg_confidence = sum(ctx.confidence for ctx in self.emotional_history) / total_samples if total_samples > 0 else 0.0

        # More nuanced logic: time-decaying average, weighting by source, detecting rapid shifts
        
        self.current_overall_mood = EmotionalContext(
            primary_emotion=dominant_mood,
            intensity=avg_intensity,
            confidence=avg_confidence,
            context_factors={"aggregation_method": "weighted_average"},
            timestamp=datetime.utcnow()
        )
        logger.debug(f"Overall mood updated: {self.current_overall_mood.primary_emotion} (Intensity: {self.current_overall_mood.intensity:.2f}, Confidence: {self.current_overall_mood.confidence:.2f})")

    async def get_suggested_response(self, current_context: Dict[str, Any], ethical_evaluation: Optional[Dict[str, Any]] = None) -> EmpathyResponse:
        """
        Suggests an empathetic response based on the current overall mood, context, and ethical evaluation.
        This builds the "interaction models that adapt based on emotional and environmental states".
        """
        logger.info(f"EmpathyEngine: Generating suggested response for context: {current_context}")
        
        overall_mood = self.current_overall_mood or EmotionalContext("neutral", 0.0, 0.0, {}, datetime.utcnow())

        response_tone = "balanced_helpful"
        emotional_acknowledgment = self.empathy_strategies["neutral"]["acknowledgment"]
        suggested_actions = self.empathy_strategies["neutral"]["actions"].copy()
        reasoning = f"Based on current overall mood: {overall_mood.primary_emotion} (Intensity: {overall_mood.intensity:.2f}, Confidence: {overall_mood.confidence:.2f})."

        # Apply strategy based on dominant mood
        strategy = self.empathy_strategies.get(overall_mood.primary_emotion, self.empathy_strategies["neutral"])
        response_tone = strategy["tone"]
        emotional_acknowledgment = strategy["acknowledgment"]
        suggested_actions = strategy["actions"].copy()

        # Adjust based on emotional intensity
        intensity_factor = overall_mood.intensity
        if intensity_factor > 0.8:
            suggested_actions.append("provide_immediate_support")
            reasoning += " High emotional intensity detected."
        
        # Incorporate contextual factors from overall_mood.context_factors
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

        # Incorporate ethical considerations if provided
        if ethical_evaluation and not ethical_evaluation.get("is_compliant", True):
            response_tone = "cautionary"
            suggested_actions.insert(0, "reconsider_action_due_to_ethical_concerns") # Prioritize ethical action
            reasoning += f" Ethical concerns raised: {ethical_evaluation.get('violation_details')}. Reconsidering action."
            emotional_acknowledgment = "I must highlight some ethical considerations here."

        # Analyze emotional trend from history for deeper context
        recent_emotions = self._analyze_emotional_trend()
        if len(recent_emotions) > 2 and all(e in ["sadness", "fear", "anger"] for e in recent_emotions[-3:]):
            suggested_actions.append("suggest_break_or_support")
            reasoning += " Persistent negative emotional trend detected."
            emotional_acknowledgment += " It seems you've been experiencing some challenging emotions lately."

        # Calculate response confidence based on overall mood confidence and other factors
        response_confidence = min(
            overall_mood.confidence * (1.0 + overall_mood.intensity * 0.2) * (1.0 if len(self.emotional_history) > 5 else 0.8),
            1.0
        )
        
        return EmpathyResponse(
            response_tone=response_tone,
            emotional_acknowledgment=emotional_acknowledgment,
            suggested_actions=suggested_actions[:5], # Limit to top 5 actions
            confidence=response_confidence,
            reasoning=reasoning
        )
        
    def _analyze_emotional_trend(self) -> List[str]:
        """Analyze recent emotional trends from history."""
        if len(self.emotional_history) < 2:
            return []
        
        # Look at the primary emotions of the last few contexts
        recent_emotions = [ctx.primary_emotion for ctx in self.emotional_history[-settings.EMOTIONAL_TREND_WINDOW:]]
        return recent_emotions
    
    def _generate_contextual_acknowledgment(
        self, 
        emotional_context: EmotionalContext, 
        base_acknowledgment: str,
        recent_emotions: List[str]
    ) -> str:
        """Generate contextual emotional acknowledgment."""
        # This method is now largely superseded by the more comprehensive get_suggested_response
        # but kept for compatibility if external calls still use it.
        # Its logic is now integrated directly into get_suggested_response.
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
        # This method is now largely superseded by the more comprehensive get_suggested_response
        # but kept for compatibility if external calls still use it.
        # Its logic is now integrated directly into get_suggested_response.
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
        if not self.emotional_history:
            return {"status": "no_data"}
        
        emotions = [ctx.primary_emotion for ctx in self.emotional_history]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        avg_intensity = sum(ctx.intensity for ctx in self.emotional_history) / len(self.emotional_history)
        
        recent_trend = emotions[-3:] if len(emotions) >= 3 else emotions
        
        return {
            "total_interactions": len(self.emotional_history),
            "emotion_distribution": emotion_counts,
            "average_intensity": round(avg_intensity, 2),
            "recent_trend": recent_trend,
            "dominant_emotion": max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral",
            "last_update": self.emotional_history[-1].timestamp.isoformat() if self.emotional_history else None
        }

