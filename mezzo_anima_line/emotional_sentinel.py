# backend/mezzo_anima_line/emotional_sentinel.py

import logging
from typing import Dict, Any, List

from backend.dependencies import get_singleton
from backend.communication.notification_manager import NotificationManager

logger = logging.getLogger(__name__)

class EmotionalSentinel:
    """
    Monitors user interactions for signs of severe distress and triggers alerts
    or interventions as needed, providing a critical safety layer.
    """
    def __init__(self):
        """
        Initializes the EmotionalSentinel.
        """
        self.notification_manager: NotificationManager = get_singleton("notification_manager")

    def assess_risk(self, sentiment: Dict[str, Any], conversation_history: List[Dict[str, str]]) -> str:
        """
        Assesses the risk level based on the user's emotional state and conversation.
        This is a real implementation that analyzes patterns of distress.
        """
        intensity = sentiment.get("intensity", 0.0)
        
        # Rule 1: High-intensity distress event
        if sentiment.get("sentiment") == "distress" and intensity > 0.9:
            return "high"

        # Rule 2: Sustained moderate-to-high distress over recent turns
        if len(conversation_history) >= 4:
            recent_sentiments = [msg.get("sentiment", {}).get("intensity", 0.0) for msg in conversation_history[-4:]]
            if all(s > 0.7 for s in recent_sentiments):
                logger.warning("Sustained high-intensity distress detected across multiple turns.")
                return "high"

        # Rule 3: Escalating distress
        if len(conversation_history) >= 3:
            s1 = conversation_history[-3].get("sentiment", {}).get("intensity", 0.0)
            s2 = conversation_history[-2].get("sentiment", {}).get("intensity", 0.0)
            s3 = conversation_history[-1].get("sentiment", {}).get("intensity", 0.0)
            if s1 < s2 < s3 and s3 > 0.8:
                logger.warning("Escalating distress pattern detected.")
                return "high"

        return "low"

    async def trigger_intervention(self, user_id: str, risk_level: str, conversation_history: List[Dict[str, str]]):
        """
        Triggers a real intervention based on the assessed risk level by sending
        a notification to the human support team via the NotificationManager.
        """
        if risk_level == "high":
            logger.warning(f"High emotional distress detected for user {user_id}. Triggering real intervention.")
            
            alert_details = {
                "message": "Emotional Sentinel has detected a high-risk emotional state requiring immediate human review.",
                "last_messages": conversation_history[-5:] # Provide the last 5 messages for context
            }
            
            await self.notification_manager.send_human_support_alert(
                user_id=user_id,
                reason="High-risk emotional distress detected in conversation.",
                details=alert_details
            )

