import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from backend.config.settings import settings

logger = logging.getLogger(__name__)

class EnvironmentalSensor:
    """Base class for environmental sensors"""
    
    def __init__(self, sensor_id: str, sensor_type: str):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.last_reading = None
        self.is_active = True
    
    async def read_data(self) -> Dict[str, Any]:
        """Read sensor data - to be implemented by specific sensors"""
        raise NotImplementedError

class MockIoTSensor(EnvironmentalSensor):
    """Mock IoT sensor for development/testing"""
    
    def __init__(self, sensor_id: str, sensor_type: str):
        super().__init__(sensor_id, sensor_type)
        self.base_values = {
            "temperature": 22.0,
            "humidity": 45.0,
            "light_level": 300.0,
            "sound_level": 35.0,
            "air_quality": 85.0
        }
    
    async def read_data(self) -> Dict[str, Any]:
        """Generate mock sensor data with realistic variations"""
        import random
        
        data = {}
        for metric, base_value in self.base_values.items():
            if self.sensor_type == metric or self.sensor_type == "multi":
                # Add realistic variation
                variation = random.uniform(-0.1, 0.1) * base_value
                data[metric] = round(base_value + variation, 2)
        
        data.update({
            "sensor_id": self.sensor_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "active"
        })
        
        self.last_reading = data
        return data

class BiometricSensor(EnvironmentalSensor):
    """Mock biometric sensor for user state monitoring"""
    
    async def read_data(self) -> Dict[str, Any]:
        import random
        
        # Simulate biometric data
        data = {
            "heart_rate": random.randint(60, 100),
            "stress_level": random.uniform(0.1, 0.8),
            "attention_level": random.uniform(0.3, 1.0),
            "fatigue_level": random.uniform(0.0, 0.7),
            "sensor_id": self.sensor_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.last_reading = data
        return data

class EnvironmentalAwarenessEngine:
    """Central engine for environmental awareness and adaptation"""
    
    def __init__(self, broker):
        self.broker = broker
        self.sensors: List[EnvironmentalSensor] = []
        self.environmental_state = {}
        self.adaptation_rules = self._load_adaptation_rules()
        self.is_running = False
    
    def _load_adaptation_rules(self) -> Dict[str, Any]:
        """Load environmental adaptation rules"""
        return {
            "temperature": {
                "cold_threshold": 18.0,
                "hot_threshold": 26.0,
                "adaptations": {
                    "cold": {"confidence_modifier": 0.95, "response_speed": 1.1},
                    "hot": {"confidence_modifier": 1.05, "response_speed": 0.95}
                }
            },
            "light_level": {
                "dim_threshold": 200.0,
                "bright_threshold": 800.0,
                "adaptations": {
                    "dim": {"attention_modifier": 0.9, "processing_depth": 1.1},
                    "bright": {"attention_modifier": 1.1, "processing_depth": 0.9}
                }
            },
            "stress_level": {
                "high_threshold": 0.7,
                "adaptations": {
                    "high_stress": {"empathy_boost": 1.2, "response_gentleness": 1.3}
                }
            }
        }
    
    def add_sensor(self, sensor: EnvironmentalSensor):
        """Add a sensor to the monitoring system"""
        self.sensors.append(sensor)
        logger.info(f"Added sensor: {sensor.sensor_id} ({sensor.sensor_type})")
    
    async def start_monitoring(self):
        """Start environmental monitoring"""
        if not settings.IOT_SENSORS_ENABLED and not settings.BIOMETRIC_MONITORING_ENABLED:
            logger.info("Environmental monitoring disabled in settings")
            return
        
        self.is_running = True
        
        # Initialize mock sensors for development
        if settings.IOT_SENSORS_ENABLED:
            self.add_sensor(MockIoTSensor("room_sensor_1", "multi"))
            self.add_sensor(MockIoTSensor("ambient_sensor_1", "temperature"))
        
        if settings.BIOMETRIC_MONITORING_ENABLED:
            self.add_sensor(BiometricSensor("biometric_1", "biometric"))
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        logger.info("Environmental awareness engine started")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect data from all sensors
                sensor_data = {}
                for sensor in self.sensors:
                    if sensor.is_active:
                        data = await sensor.read_data()
                        sensor_data[sensor.sensor_id] = data
                
                # Update environmental state
                self.environmental_state.update({
                    "sensors": sensor_data,
                    "last_update": datetime.utcnow().isoformat(),
                    "adaptations": self._calculate_adaptations(sensor_data)
                })
                
                # Broadcast environmental data to agents
                await self.broker.publish_message("environmental_data", {
                    "type": "environmental_update",
                    "data": self.environmental_state,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Wait before next reading
                await asyncio.sleep(30)  # Read every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in environmental monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _calculate_adaptations(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate behavioral adaptations based on sensor data"""
        adaptations = {}
        
        # Aggregate sensor readings
        aggregated = self._aggregate_sensor_data(sensor_data)
        
        # Apply adaptation rules
        for metric, value in aggregated.items():
            if metric in self.adaptation_rules:
                rules = self.adaptation_rules[metric]
                
                if metric == "temperature":
                    if value < rules["cold_threshold"]:
                        adaptations.update(rules["adaptations"]["cold"])
                    elif value > rules["hot_threshold"]:
                        adaptations.update(rules["adaptations"]["hot"])
                
                elif metric == "light_level":
                    if value < rules["dim_threshold"]:
                        adaptations.update(rules["adaptations"]["dim"])
                    elif value > rules["bright_threshold"]:
                        adaptations.update(rules["adaptations"]["bright"])
                
                elif metric == "stress_level":
                    if value > rules["high_threshold"]:
                        adaptations.update(rules["adaptations"]["high_stress"])
        
        return adaptations
    
    def _aggregate_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate sensor data across multiple sensors"""
        aggregated = {}
        counts = {}
        
        for sensor_id, data in sensor_data.items():
            for metric, value in data.items():
                if isinstance(value, (int, float)):
                    if metric not in aggregated:
                        aggregated[metric] = 0
                        counts[metric] = 0
                    aggregated[metric] += value
                    counts[metric] += 1
        
        # Calculate averages
        for metric in aggregated:
            if counts[metric] > 0:
                aggregated[metric] = aggregated[metric] / counts[metric]
        
        return aggregated
    
    async def stop_monitoring(self):
        """Stop environmental monitoring"""
        self.is_running = False
        logger.info("Environmental awareness engine stopped")