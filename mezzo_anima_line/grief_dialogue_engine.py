# backend/mezzo_anima_line/grief_dialogue_engine.py

import logging
from typing import Dict, Any, Tuple

from backend.support_ai.llm_client import LLMClient

logger = logging.getLogger(__name__)

class GriefDialogueEngine:
    """
    A specialized conversational AI that can detect user distress, provide
    empathetic responses, and manage conversations with a focus on emotional support.
    """
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def generate_empathetic_response(self, user_input: str, conversation_history: list) -> Tuple[str, Dict[str, Any]]:
        """
        Generates an empathetic response based on the user's input and conversation history.
        """
        sentiment = await self._analyze_sentiment(user_input)
        
        prompt = self._construct_prompt(user_input, conversation_history, sentiment)
        
        response_text = await self.llm_client.generate_text(prompt)
        
        return response_text, sentiment

    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the user's input to detect distress.
        """
        # In a real system, this would use a fine-tuned sentiment analysis model
        # For now, we'll use a keyword-based approach for demonstration
        distress_keywords = ["sad", "lonely", "grieving", "miss", "hard"]
        if any(keyword in text.lower() for keyword in distress_keywords):
            return {"sentiment": "distress", "intensity": 0.8}
        else:
            return {"sentiment": "neutral", "intensity": 0.5}

    def _construct_prompt(self, user_input: str, history: list, sentiment: Dict[str, Any]) -> str:
        """
        Constructs a prompt for the LLM that encourages an empathetic and supportive response.
        """
        base_prompt = (
            "You are MEZZO, an AI companion designed for emotional support and digital legacy. "
            "Your user is expressing feelings of {sentiment}. "
            "Respond with empathy, compassion, and understanding. Avoid giving unsolicited advice. "
            "Focus on validating their feelings and offering a supportive presence."
        ).format(sentiment=sentiment['sentiment'])

        # Add conversation history for context
        for message in history:
            base_prompt += f"\n{message['role']}: {message['content']}"
        
        base_prompt += f"\nUser: {user_input}\nMEZZO:"
        return base_prompt
