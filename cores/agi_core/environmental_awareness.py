# backend/agi_core/environmental_awareness.py

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

class EnvironmentalSensor(BaseModel):
    """Abstract base model for a sensor."""
    sensor_id: str
    sensor_type: str  # e.g., 'temperature', 'humidity'

class MockIoTSensor(EnvironmentalSensor):
    """A mock IoT sensor for simulation."""
    def read_value(self) -> float:
        if self.sensor_type == 'temperature':
            return 22.5
        elif self.sensor_type == 'humidity':
            return 55.0
        return 0.0

class EnvironmentalAwarenessEngine:
    """
    Manages sensory inputs and adapts AI behavior based on environmental context.
    This version is enhanced to handle potentially missing or null data from real sensors.
    """
    def __init__(self):
        self.sensors: Dict[str, EnvironmentalSensor] = {}
        self.current_state: Dict[str, Any] = {}
        self.behavioral_rules = {
            "temperature_threshold_high": 28.0,
            "humidity_threshold_high": 70.0,
        }
        logger.info("EnvironmentalAwarenessEngine initialized.")

    def register_sensor(self, sensor: EnvironmentalSensor):
        """Registers a new sensor with the engine."""
        self.sensors[sensor.sensor_id] = sensor
        logger.info(f"Registered sensor: {sensor.sensor_id} ({sensor.sensor_type})")

    def update_state_from_data(self, data: Dict[str, Any]):
        """
        Updates the engine's current state from a dictionary of sensory data.
        Handles potential errors and missing values gracefully.
        """
        if not isinstance(data, dict):
            logger.warning(f"Received invalid data for state update: {data}")
            return

        # Safely update state, checking for None values
        self.current_state['temperature'] = data.get('temperature_celsius')
        self.current_state['humidity'] = data.get('humidity_percent')
        self.current_state['light_level'] = data.get('light_lux')
        self.current_state['sound_level'] = data.get('sound_db')
        self.current_state['last_updated'] = data.get('timestamp')
        
        logger.info(f"Environmental state updated: {self.current_state}")

    def get_contextual_recommendation(self) -> Optional[str]:
        """
        Provides a behavioral recommendation based on the current environmental state.
        This logic is now more robust against missing data.
        """
        temp = self.current_state.get('temperature')
        humidity = self.current_state.get('humidity')

        if temp is not None and temp > self.behavioral_rules["temperature_threshold_high"]:
            logger.warning(f"High temperature detected: {temp}°C. Recommending caution.")
            return "ADJUST_BEHAVIOR_CAUTIOUS"
        
        if humidity is not None and humidity > self.behavioral_rules["humidity_threshold_high"]:
            logger.warning(f"High humidity detected: {humidity}%. Recommending reduced activity.")
            return "ADJUST_BEHAVIOR_REDUCED_ACTIVITY"
            
        logger.info("Environmental conditions are within normal parameters.")
        return None

