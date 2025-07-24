import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import re

from backend.config.settings import settings

logger = logging.getLogger(__name__)

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

class EmotionDetector:
    """Advanced emotion detection from text and context"""
    
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
    
    def detect_emotion(self, text: str, context: Dict[str, Any] = None) -> EmotionalContext:
        """Detect emotion from text with contextual awareness"""
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
    
    def _analyze_context(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contextual factors that might influence emotion"""
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
    """Advanced empathy engine for emotional intelligence"""
    
    def __init__(self, broker):
        self.broker = broker
        self.emotion_detector = EmotionDetector()
        self.emotional_history: List[EmotionalContext] = []
        self.empathy_strategies = self._load_empathy_strategies()
        self.is_active = settings.EMPATHY_ENGINE_ENABLED
    
    def _load_empathy_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load empathy response strategies for different emotions"""
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
            "neutral": {
                "tone": "balanced_helpful",
                "acknowledgment": "I'm here to help you with whatever you need.",
                "actions": ["assess_needs", "provide_information", "maintain_engagement"]
            }
        }
    
    async def analyze_emotional_context(self, text: str, user_context: Dict[str, Any] = None) -> EmotionalContext:
        """Analyze emotional context of user input"""
        if not self.is_active:
            return EmotionalContext("neutral", 0.5, 0.3, {}, datetime.utcnow())
        
        emotional_context = self.emotion_detector.detect_emotion(text, user_context)
        
        # Store in emotional history
        self.emotional_history.append(emotional_context)
        
        # Keep only recent history
        cutoff_time = datetime.utcnow() - timedelta(minutes=settings.EMOTIONAL_CONTEXT_WINDOW)
        self.emotional_history = [
            ctx for ctx in self.emotional_history 
            if ctx.timestamp > cutoff_time
        ]
        
        return emotional_context
    
    def generate_empathic_response(self, emotional_context: EmotionalContext) -> EmpathyResponse:
        """Generate empathic response based on emotional context"""
        if not self.is_active:
            return EmpathyResponse("neutral", "", [], 0.5)
        
        emotion = emotional_context.primary_emotion
        strategy = self.empathy_strategies.get(emotion, self.empathy_strategies["neutral"])
        
        # Adjust response based on emotional intensity
        intensity_factor = emotional_context.intensity
        
        # Consider emotional history for context
        recent_emotions = self._analyze_emotional_trend()
        
        # Generate contextual acknowledgment
        acknowledgment = self._generate_contextual_acknowledgment(
            emotional_context, strategy["acknowledgment"], recent_emotions
        )
        
        # Determine appropriate actions
        actions = self._select_appropriate_actions(
            emotional_context, strategy["actions"], recent_emotions
        )
        
        # Calculate response confidence
        confidence = min(
            emotional_context.confidence * 
            (1.0 + intensity_factor * 0.3) * 
            (1.0 if len(recent_emotions) > 1 else 0.8),
            1.0
        )
        
        return EmpathyResponse(
            response_tone=strategy["tone"],
            emotional_acknowledgment=acknowledgment,
            suggested_actions=actions,
            confidence=confidence
        )
    
    def _analyze_emotional_trend(self) -> List[str]:
        """Analyze recent emotional trends"""
        if len(self.emotional_history) < 2:
            return []
        
        recent_emotions = [ctx.primary_emotion for ctx in self.emotional_history[-3:]]
        return recent_emotions
    
    def _generate_contextual_acknowledgment(
        self, 
        emotional_context: EmotionalContext, 
        base_acknowledgment: str,
        recent_emotions: List[str]
    ) -> str:
        """Generate contextual emotional acknowledgment"""
        acknowledgment = base_acknowledgment
        
        # Add context based on factors
        if "financial" in emotional_context.context_factors.get("domain", ""):
            acknowledgment += " I understand this involves your financial well-being."
        
        if emotional_context.context_factors.get("urgency") == "high":
            acknowledgment += " I can see this is time-sensitive for you."
        
        if "environmental_stress" in emotional_context.context_factors:
            acknowledgment += " I notice you might be dealing with additional stress right now."
        
        # Consider emotional progression
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
        """Select appropriate empathic actions"""
        actions = base_actions.copy()
        
        # Add context-specific actions
        if emotional_context.intensity > 0.8:
            actions.append("provide_immediate_support")
        
        if "financial" in emotional_context.context_factors.get("domain", ""):
            actions.append("offer_financial_guidance")
        
        if emotional_context.context_factors.get("urgency") == "high":
            actions.append("prioritize_response_speed")
        
        # Consider emotional history
        if len(recent_emotions) > 2 and all(e in ["sadness", "fear", "anger"] for e in recent_emotions[-3:]):
            actions.append("suggest_break_or_support")
        
        return actions[:5]  # Limit to top 5 actions
    
    async def adapt_communication_style(self, empathy_response: EmpathyResponse) -> Dict[str, Any]:
        """Adapt communication style based on empathic analysis"""
        adaptations = {
            "tone": empathy_response.response_tone,
            "pace": "normal",
            "detail_level": "standard",
            "supportiveness": "standard"
        }
        
        # Adjust based on response tone
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
        """Get insights from emotional interaction history"""
        if not self.emotional_history:
            return {"status": "no_data"}
        
        # Analyze emotional patterns
        emotions = [ctx.primary_emotion for ctx in self.emotional_history]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate average intensity
        avg_intensity = sum(ctx.intensity for ctx in self.emotional_history) / len(self.emotional_history)
        
        # Identify trends
        recent_trend = emotions[-3:] if len(emotions) >= 3 else emotions
        
        return {
            "total_interactions": len(self.emotional_history),
            "emotion_distribution": emotion_counts,
            "average_intensity": round(avg_intensity, 2),
            "recent_trend": recent_trend,
            "dominant_emotion": max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral",
            "last_update": self.emotional_history[-1].timestamp.isoformat() if self.emotional_history else None
        }