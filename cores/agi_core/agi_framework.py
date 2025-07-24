import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from backend.config.settings import settings, AGIPhase
from backend.multi_agent.message_broker_interface import MessageBrokerInterface

logger = logging.getLogger(__name__)

class AGICapability(str, Enum):
    ADAPTIVE_LEARNING = "adaptive_learning"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    SELF_HEALING = "self_healing"
    ETHICAL_REASONING = "ethical_reasoning"
    ENVIRONMENTAL_AWARENESS = "environmental_awareness"
    COLLECTIVE_INTELLIGENCE = "collective_intelligence"

@dataclass
class AGIState:
    phase: AGIPhase
    active_capabilities: List[AGICapability]
    learning_rate: float
    emotional_state: Dict[str, float]
    environmental_context: Dict[str, Any]
    collective_knowledge: Dict[str, Any]
    last_update: datetime

class AGICore(ABC):
    """Base class for AGI-enabled agents with advanced capabilities"""
    
    def __init__(self, agent_id: str, broker: MessageBrokerInterface):
        self.agent_id = agent_id
        self.broker = broker
        self.agi_state = AGIState(
            phase=settings.CURRENT_AGI_PHASE,
            active_capabilities=[],
            learning_rate=0.1,
            emotional_state={"confidence": 0.5, "empathy": 0.5, "curiosity": 0.5},
            environmental_context={},
            collective_knowledge={},
            last_update=datetime.utcnow()
        )
        self.capability_modules = {}
        self._initialize_capabilities()
    
    def _initialize_capabilities(self):
        """Initialize AGI capabilities based on current phase"""
        if settings.ADAPTIVE_LEARNING_ENABLED:
            self.agi_state.active_capabilities.append(AGICapability.ADAPTIVE_LEARNING)
        
        if settings.EMPATHY_ENGINE_ENABLED:
            self.agi_state.active_capabilities.append(AGICapability.EMOTIONAL_INTELLIGENCE)
        
        if settings.SELF_HEALING_ENABLED:
            self.agi_state.active_capabilities.append(AGICapability.SELF_HEALING)
        
        if settings.ENVIRONMENTAL_ADAPTATION_ENABLED:
            self.agi_state.active_capabilities.append(AGICapability.ENVIRONMENTAL_AWARENESS)
        
        if settings.COLLECTIVE_INTELLIGENCE_ENABLED:
            self.agi_state.active_capabilities.append(AGICapability.COLLECTIVE_INTELLIGENCE)
    
    async def evolve_capabilities(self):
        """Evolve AGI capabilities based on experience and feedback"""
        for capability in self.agi_state.active_capabilities:
            await self._evolve_capability(capability)
        
        self.agi_state.last_update = datetime.utcnow()
    
    @abstractmethod
    async def _evolve_capability(self, capability: AGICapability):
        """Implement capability-specific evolution logic"""
        pass
    
    async def adapt_to_environment(self, environmental_data: Dict[str, Any]):
        """Adapt behavior based on environmental context"""
        if AGICapability.ENVIRONMENTAL_AWARENESS not in self.agi_state.active_capabilities:
            return
        
        self.agi_state.environmental_context.update(environmental_data)
        
        # Adjust emotional state based on environment
        if "temperature" in environmental_data:
            temp = environmental_data["temperature"]
            if temp < 18:  # Cold environment
                self.agi_state.emotional_state["confidence"] *= 0.95
            elif temp > 26:  # Warm environment
                self.agi_state.emotional_state["confidence"] *= 1.05
        
        logger.info(f"AGI {self.agent_id}: Adapted to environment - {environmental_data}")
    
    async def share_collective_knowledge(self, knowledge: Dict[str, Any]):
        """Share knowledge with the collective intelligence network"""
        if AGICapability.COLLECTIVE_INTELLIGENCE not in self.agi_state.active_capabilities:
            return
        
        message = {
            "type": "collective_knowledge_share",
            "agent_id": self.agent_id,
            "knowledge": knowledge,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broker.publish_message("collective_intelligence", message)
    
    async def self_heal(self, error_context: Dict[str, Any]):
        """Attempt to self-heal from errors or degraded performance"""
        if AGICapability.SELF_HEALING not in self.agi_state.active_capabilities:
            return False
        
        healing_strategies = [
            self._reset_emotional_state,
            self._adjust_learning_rate,
            self._clear_corrupted_memory,
            self._request_collective_assistance
        ]
        
        for strategy in healing_strategies:
            try:
                success = await strategy(error_context)
                if success:
                    logger.info(f"AGI {self.agent_id}: Self-healing successful using {strategy.__name__}")
                    return True
            except Exception as e:
                logger.warning(f"AGI {self.agent_id}: Healing strategy {strategy.__name__} failed: {e}")
        
        return False
    
    async def _reset_emotional_state(self, error_context: Dict[str, Any]) -> bool:
        """Reset emotional state to baseline"""
        self.agi_state.emotional_state = {"confidence": 0.5, "empathy": 0.5, "curiosity": 0.5}
        return True
    
    async def _adjust_learning_rate(self, error_context: Dict[str, Any]) -> bool:
        """Adjust learning rate based on error patterns"""
        if "learning_instability" in error_context:
            self.agi_state.learning_rate *= 0.8
            return True
        return False
    
    async def _clear_corrupted_memory(self, error_context: Dict[str, Any]) -> bool:
        """Clear potentially corrupted memory segments"""
        if "memory_corruption" in error_context:
            self.agi_state.collective_knowledge.clear()
            return True
        return False
    
    async def _request_collective_assistance(self, error_context: Dict[str, Any]) -> bool:
        """Request assistance from collective intelligence"""
        message = {
            "type": "healing_assistance_request",
            "agent_id": self.agent_id,
            "error_context": error_context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broker.publish_message("collective_intelligence", message)
        return True