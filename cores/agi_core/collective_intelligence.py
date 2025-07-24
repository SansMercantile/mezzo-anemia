import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib

from backend.config.settings import settings

logger = logging.getLogger(__name__)

class KnowledgeType(str, Enum):
    MARKET_INSIGHT = "market_insight"
    TECHNICAL_PATTERN = "technical_pattern"
    RISK_ASSESSMENT = "risk_assessment"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    ENVIRONMENTAL_ADAPTATION = "environmental_adaptation"
    ERROR_RESOLUTION = "error_resolution"
    STRATEGY_OPTIMIZATION = "strategy_optimization"

@dataclass
class KnowledgeItem:
    id: str
    type: KnowledgeType
    content: Dict[str, Any]
    source_agent: str
    confidence: float
    validation_count: int
    creation_time: datetime
    last_accessed: datetime
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "creation_time": self.creation_time.isoformat(),
            "last_accessed": self.last_accessed.isoformat()
        }

@dataclass
class ConsensusRequest:
    id: str
    question: str
    context: Dict[str, Any]
    requesting_agent: str
    responses: Dict[str, Any]  # agent_id -> response
    deadline: datetime
    min_responses: int
    
class CollectiveIntelligenceEngine:
    """Engine for managing collective intelligence across AGI agents"""
    
    def __init__(self, broker):
        self.broker = broker
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
        self.active_consensus_requests: Dict[str, ConsensusRequest] = {}
        self.agent_reputation: Dict[str, float] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        self.is_active = settings.COLLECTIVE_INTELLIGENCE_ENABLED
        self.knowledge_validation_threshold = 3
    
    async def start(self):
        """Start the collective intelligence engine"""
        if not self.is_active:
            logger.info("Collective Intelligence disabled in settings")
            return
        
        # Subscribe to collective intelligence topics
        await self.broker.subscribe_to_topic(
            "collective_knowledge_share", 
            self._handle_knowledge_share,
            "collective-intelligence-knowledge"
        )
        
        await self.broker.subscribe_to_topic(
            "collective_intelligence",
            self._handle_collective_message,
            "collective-intelligence-main"
        )
        
        # Start maintenance tasks
        asyncio.create_task(self._maintenance_loop())
        
        logger.info("Collective Intelligence Engine started")
    
    async def _handle_knowledge_share(self, message_payload: Dict[str, Any]):
        """Handle knowledge sharing from agents"""
        try:
            payload = message_payload.get('payload', {})
            agent_id = payload.get('agent_id')
            knowledge = payload.get('knowledge', {})
            
            if not agent_id or not knowledge:
                return
            
            # Create knowledge item
            knowledge_item = KnowledgeItem(
                id=self._generate_knowledge_id(knowledge, agent_id),
                type=KnowledgeType(knowledge.get('type', 'market_insight')),
                content=knowledge,
                source_agent=agent_id,
                confidence=knowledge.get('confidence', 0.5),
                validation_count=1,
                creation_time=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                tags=knowledge.get('tags', [])
            )
            
            # Store knowledge
            await self._store_knowledge(knowledge_item)
            
            # Update agent reputation
            self._update_agent_reputation(agent_id, "knowledge_share", 0.1)
            
            logger.info(f"Stored knowledge from {agent_id}: {knowledge_item.type}")
            
        except Exception as e:
            logger.error(f"Error handling knowledge share: {e}")
    
    async def _handle_collective_message(self, message_payload: Dict[str, Any]):
        """Handle various collective intelligence messages"""
        try:
            payload = message_payload.get('payload', {})
            message_type = payload.get('type')
            
            if message_type == "consensus_request":
                await self._handle_consensus_request(payload)
            elif message_type == "consensus_response":
                await self._handle_consensus_response(payload)
            elif message_type == "healing_assistance_request":
                await self._handle_healing_assistance(payload)
            elif message_type == "knowledge_query":
                await self._handle_knowledge_query(payload)
                
        except Exception as e:
            logger.error(f"Error handling collective message: {e}")
    
    async def _store_knowledge(self, knowledge_item: KnowledgeItem):
        """Store knowledge item in the collective knowledge base"""
        # Check if similar knowledge already exists
        existing_id = await self._find_similar_knowledge(knowledge_item)
        
        if existing_id:
            # Update existing knowledge
            existing = self.knowledge_base[existing_id]
            existing.validation_count += 1
            existing.confidence = min(
                (existing.confidence + knowledge_item.confidence) / 2 * 1.1,
                1.0
            )
            existing.last_accessed = datetime.utcnow()
            logger.info(f"Updated existing knowledge: {existing_id}")
        else:
            # Store new knowledge
            self.knowledge_base[knowledge_item.id] = knowledge_item
            logger.info(f"Stored new knowledge: {knowledge_item.id}")
    
    async def _find_similar_knowledge(self, knowledge_item: KnowledgeItem) -> Optional[str]:
        """Find similar knowledge in the knowledge base"""
        for existing_id, existing_item in self.knowledge_base.items():
            if (existing_item.type == knowledge_item.type and
                self._calculate_similarity(existing_item.content, knowledge_item.content) > 0.8):
                return existing_id
        return None
    
    def _calculate_similarity(self, content1: Dict[str, Any], content2: Dict[str, Any]) -> float:
        """Calculate similarity between two knowledge contents"""
        # Simple similarity based on common keys and values
        keys1 = set(content1.keys())
        keys2 = set(content2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        common_keys = keys1.intersection(keys2)
        similarity = len(common_keys) / len(keys1.union(keys2))
        
        # Check value similarity for common keys
        value_similarity = 0.0
        for key in common_keys:
            if content1[key] == content2[key]:
                value_similarity += 1.0
        
        if common_keys:
            value_similarity /= len(common_keys)
        
        return (similarity + value_similarity) / 2
    
    async def request_consensus(
        self, 
        question: str, 
        context: Dict[str, Any], 
        requesting_agent: str,
        min_responses: int = 3,
        timeout_minutes: int = 5
    ) -> str:
        """Request consensus from the collective"""
        request_id = self._generate_request_id(question, requesting_agent)
        
        consensus_request = ConsensusRequest(
            id=request_id,
            question=question,
            context=context,
            requesting_agent=requesting_agent,
            responses={},
            deadline=datetime.utcnow() + timedelta(minutes=timeout_minutes),
            min_responses=min_responses
        )
        
        self.active_consensus_requests[request_id] = consensus_request
        
        # Broadcast consensus request
        message = {
            "type": "consensus_request",
            "request_id": request_id,
            "question": question,
            "context": context,
            "requesting_agent": requesting_agent,
            "deadline": consensus_request.deadline.isoformat(),
            "min_responses": min_responses
        }
        
        await self.broker.publish_message("collective_intelligence", message)
        
        logger.info(f"Consensus request {request_id} initiated by {requesting_agent}")
        return request_id
    
    async def _handle_consensus_request(self, payload: Dict[str, Any]):
        """Handle incoming consensus requests"""
        request_id = payload.get('request_id')
        question = payload.get('question')
        context = payload.get('context', {})
        requesting_agent = payload.get('requesting_agent')
        
        if not all([request_id, question, requesting_agent]):
            return
        
        # Generate collective response based on knowledge base
        response = await self._generate_consensus_response(question, context)
        
        # Send response
        response_message = {
            "type": "consensus_response",
            "request_id": request_id,
            "response": response,
            "responding_agent": "collective_intelligence_engine",
            "confidence": response.get('confidence', 0.5)
        }
        
        await self.broker.publish_message("collective_intelligence", response_message)
    
    async def _generate_consensus_response(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on collective knowledge"""
        relevant_knowledge = await self._query_knowledge(question, context)
        
        if not relevant_knowledge:
            return {
                "answer": "Insufficient collective knowledge to provide consensus",
                "confidence": 0.1,
                "knowledge_items_used": 0
            }
        
        # Aggregate knowledge for consensus
        aggregated_insights = []
        total_confidence = 0.0
        
        for knowledge_item in relevant_knowledge:
            aggregated_insights.append(knowledge_item.content)
            total_confidence += knowledge_item.confidence * (knowledge_item.validation_count / 10)
        
        avg_confidence = min(total_confidence / len(relevant_knowledge), 1.0)
        
        return {
            "answer": f"Based on {len(relevant_knowledge)} knowledge items from the collective",
            "insights": aggregated_insights[:5],  # Top 5 insights
            "confidence": avg_confidence,
            "knowledge_items_used": len(relevant_knowledge)
        }
    
    async def _query_knowledge(self, query: str, context: Dict[str, Any]) -> List[KnowledgeItem]:
        """Query the knowledge base for relevant information"""
        relevant_items = []
        query_lower = query.lower()
        
        for knowledge_item in self.knowledge_base.values():
            # Simple relevance scoring
            relevance_score = 0.0
            
            # Check content relevance
            content_str = json.dumps(knowledge_item.content).lower()
            if any(word in content_str for word in query_lower.split()):
                relevance_score += 0.5
            
            # Check tag relevance
            for tag in knowledge_item.tags:
                if tag.lower() in query_lower:
                    relevance_score += 0.3
            
            # Consider validation count and confidence
            relevance_score *= (knowledge_item.confidence * 
                              min(knowledge_item.validation_count / 5, 1.0))
            
            if relevance_score > 0.2:
                relevant_items.append(knowledge_item)
        
        # Sort by relevance and return top items
        relevant_items.sort(key=lambda x: x.confidence * x.validation_count, reverse=True)
        return relevant_items[:10]
    
    async def _handle_healing_assistance(self, payload: Dict[str, Any]):
        """Handle healing assistance requests"""
        agent_id = payload.get('agent_id')
        error_context = payload.get('error_context', {})
        
        if not agent_id:
            return
        
        # Query knowledge base for similar error resolutions
        healing_knowledge = []
        for knowledge_item in self.knowledge_base.values():
            if (knowledge_item.type == KnowledgeType.ERROR_RESOLUTION and
                knowledge_item.confidence > 0.6):
                healing_knowledge.append(knowledge_item)
        
        # Send healing suggestions
        healing_response = {
            "type": "healing_assistance_response",
            "target_agent": agent_id,
            "suggestions": [item.content for item in healing_knowledge[:3]],
            "confidence": sum(item.confidence for item in healing_knowledge) / max(len(healing_knowledge), 1)
        }
        
        await self.broker.publish_message("collective_intelligence", healing_response)
        
        logger.info(f"Provided healing assistance to {agent_id}")
    
    def _update_agent_reputation(self, agent_id: str, action: str, delta: float):
        """Update agent reputation based on actions"""
        if agent_id not in self.agent_reputation:
            self.agent_reputation[agent_id] = 0.5
        
        current_rep = self.agent_reputation[agent_id]
        
        if action == "knowledge_share":
            new_rep = min(current_rep + delta, 1.0)
        elif action == "consensus_participation":
            new_rep = min(current_rep + delta * 0.5, 1.0)
        elif action == "healing_assistance":
            new_rep = min(current_rep + delta * 1.5, 1.0)
        else:
            new_rep = current_rep
        
        self.agent_reputation[agent_id] = new_rep
    
    async def _maintenance_loop(self):
        """Periodic maintenance of the knowledge base"""
        while self.is_active:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up old consensus requests
                current_time = datetime.utcnow()
                expired_requests = [
                    req_id for req_id, req in self.active_consensus_requests.items()
                    if req.deadline < current_time
                ]
                
                for req_id in expired_requests:
                    del self.active_consensus_requests[req_id]
                
                # Archive old knowledge items
                cutoff_time = current_time - timedelta(days=7)
                archived_count = 0
                
                for knowledge_id, knowledge_item in list(self.knowledge_base.items()):
                    if (knowledge_item.last_accessed < cutoff_time and 
                        knowledge_item.validation_count < 2):
                        del self.knowledge_base[knowledge_id]
                        archived_count += 1
                
                if archived_count > 0:
                    logger.info(f"Archived {archived_count} old knowledge items")
                
            except Exception as e:
                logger.error(f"Error in collective intelligence maintenance: {e}")
    
    def _generate_knowledge_id(self, knowledge: Dict[str, Any], agent_id: str) -> str:
        """Generate unique ID for knowledge item"""
        content_hash = hashlib.md5(
            json.dumps(knowledge, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{agent_id}_{content_hash}_{int(datetime.utcnow().timestamp())}"
    
    def _generate_request_id(self, question: str, agent_id: str) -> str:
        """Generate unique ID for consensus request"""
        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        return f"consensus_{agent_id}_{question_hash}_{int(datetime.utcnow().timestamp())}"
    
    def get_collective_stats(self) -> Dict[str, Any]:
        """Get statistics about the collective intelligence"""
        return {
            "total_knowledge_items": len(self.knowledge_base),
            "active_consensus_requests": len(self.active_consensus_requests),
            "agent_count": len(self.agent_reputation),
            "knowledge_by_type": {
                ktype.value: sum(1 for item in self.knowledge_base.values() if item.type == ktype)
                for ktype in KnowledgeType
            },
            "average_agent_reputation": (
                sum(self.agent_reputation.values()) / len(self.agent_reputation)
                if self.agent_reputation else 0.0
            ),
            "last_activity": max(
                (item.last_accessed for item in self.knowledge_base.values()),
                default=datetime.utcnow()
            ).isoformat()
        }