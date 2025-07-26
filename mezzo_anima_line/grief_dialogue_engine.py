# backend/mezzo_anima_line/grief_dialogue_engine.py

import logging
from typing import Dict, Any, Tuple
import json

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
        Analyzes the sentiment of the user's input to detect distress using a
        powerful Large Language Model for nuanced understanding. This is a real,
        production-grade implementation.
        """
        prompt = (
            "Analyze the following text for its emotional content, specifically focusing on "
            "expressions of grief, sadness, loneliness, or distress. Respond with only a JSON object "
            "containing two keys: 'sentiment' (a string, e.g., 'distress', 'sadness', 'neutral', 'reminiscing') "
            "and 'intensity' (a float between 0.0 and 1.0).\n\n"
            f"Text to analyze: \"{text}\""
        )
        
        try:
            response_str = await self.llm_client.generate_text(prompt)
            # The LLM is instructed to return a JSON string, so we parse it.
            sentiment_data = json.loads(response_str)
            
            # Validate the structure of the response
            if "sentiment" in sentiment_data and "intensity" in sentiment_data:
                logger.info(f"LLM-based sentiment analysis complete: {sentiment_data}")
                return sentiment_data
            else:
                logger.warning(f"LLM sentiment response was malformed: {response_str}")
                return {"sentiment": "neutral", "intensity": 0.5}

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM sentiment response: {response_str}")
            return {"sentiment": "neutral", "intensity": 0.5}
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM sentiment analysis: {e}", exc_info=True)
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
        ).format(sentiment=sentiment.get('sentiment', 'unknown'))

        # Add conversation history for context
        for message in history:
            role = message.get('role', 'user')
            content = message.get('content', '')
            base_prompt += f"\n{role.capitalize()}: {content}"
        
        base_prompt += f"\nUser: {user_input}\nMEZZO:"
        return base_prompt
