# sans-mercantile-app/backend/mezzo_anima_line/mezzo_api.py
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File # Added UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

# Import Mezzo's specific orchestrator and agent protocol
from backend.mezzo_anima_line.mezzo_orchestrator import MezzoOrchestrator
from backend.mezzo_anima_line.mezzo_agent_protocol import MezzoAgentType, MezzoAgentMessage

# Import new dependencies for Digital Legacy and Grief Dialogue
from backend.mezzo_anima_line.digital_legacy_manager import DigitalLegacyManager
from backend.mezzo_anima_line.grief_dialogue_engine import GriefDialogueEngine

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to get the MezzoOrchestrator instance
from backend.mezzo_anima_line.main import mezzo_orchestrator_instance # Assuming main.py exposes this

def get_mezzo_orchestrator() -> MezzoOrchestrator:
    """
    Dependency to get the running instance of Mezzo's Orchestrator.
    """
    if mezzo_orchestrator_instance and mezzo_orchestrator_instance.is_running:
        return mezzo_orchestrator_instance
    raise HTTPException(status_code=503, detail="Mezzo Orchestrator not initialized or running.")

# Dependencies for Digital Legacy Manager and Grief Dialogue Engine
# These assume these managers/engines are instantiated as singletons or
# are accessible via MezzoOrchestrator and exposed through a dependency function.
# For now, we'll create placeholder dependency functions that you'll link to your
# actual singleton management in backend/dependencies.py or mezzo_anima_line/main.py.

def get_digital_legacy_manager_dependency() -> DigitalLegacyManager:
    """Dependency to get the DigitalLegacyManager instance."""
    # This should return the actual singleton instance of DigitalLegacyManager
    # from your dependencies.py or from mezzo_orchestrator_instance.
    from backend.dependencies import get_digital_legacy_manager 
    return get_digital_legacy_manager()

def get_grief_dialogue_engine_dependency() -> GriefDialogueEngine:
    """Dependency to get the GriefDialogueEngine instance."""
    # This should return the actual singleton instance of GriefDialogueEngine
    # from your dependencies.py or from mezzo_orchestrator_instance.
    from backend.dependencies import get_grief_dialogue_engine 
    return get_grief_dialogue_engine()


class MezzoTaskRequest(BaseModel):
    agent_id: str
    task_type: str
    task_data: Dict[str, Any]

class MezzoConverseRequest(BaseModel):
    user_id: str
    agent_id: str # Target Mezzo agent (e.g., MezzoMaterna-childID)
    message: str
    user_context: Optional[Dict[str, Any]] = None
    audio_features: Optional[Dict[str, Any]] = None
    vision_features: Optional[Dict[str, Any]] = None
    system_status: Optional[Dict[str, Any]] = None
    environmental_data: Optional[Dict[str, Any]] = None


@router.post("/mezzo/task", status_code=202)
async def route_mezzo_task(
    request: MezzoTaskRequest,
    orchestrator: MezzoOrchestrator = Depends(get_mezzo_orchestrator)
):
    """
    Routes a task to a specific Mezzo agent.
    """
    agent = orchestrator.agents.get(request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Mezzo Agent {request.agent_id} not found.")
    
    mezzo_message = MezzoAgentMessage(
        sender_id="api_gateway",
        sender_type=MezzoAgentType.EXTERNAL_API,
        recipient_id=request.agent_id,
        message_type=request.task_type,
        content=request.task_data
    )
    await agent.task_queue.put(mezzo_message)
    return {"message": f"Task '{request.task_type}' routed to Mezzo agent {request.agent_id}."}

@router.post("/mezzo/converse", status_code=200)
async def mezzo_converse(
    request: MezzoConverseRequest,
    orchestrator: MezzoOrchestrator = Depends(get_mezzo_orchestrator)
):
    """
    Initiates or continues a conversation with a specific Mezzo agent,
    leveraging its emotional intelligence and memory.
    """
    agent = orchestrator.agents.get(request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Mezzo Agent {request.agent_id} not found.")

    try:
        response_content = await agent.converse(
            user_id=request.user_id,
            message=request.message,
            user_context=request.user_context,
            audio_features=request.audio_features,
            vision_features=request.vision_features,
            system_status=request.system_status,
            environmental_data=request.environmental_data
        )
        return {"response": response_content}
    except Exception as e:
        logger.error(f"Error during Mezzo conversation with agent {request.agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing conversation: {e}")

@router.get("/mezzo/agents", response_model=List[Dict[str, Any]])
async def get_mezzo_agents(
    orchestrator: MezzoOrchestrator = Depends(get_mezzo_orchestrator)
):
    """
    Retrieves a list of all managed Mezzo agents.
    """
    return [
        {"agent_id": agent.agent_id, "agent_type": agent.agent_type.value, "status": agent.state.value}
        for agent in orchestrator.agents.values()
    ]

# --- New Endpoints for Digital Legacy and Grief Dialogue ---

@router.post("/mezzo/legacy/document", status_code=201)
async def upload_legal_document(
    user_id: str,
    file: UploadFile = File(...),
    manager: DigitalLegacyManager = Depends(get_digital_legacy_manager_dependency)
):
    """
    Uploads a legal document for secure storage via the DigitalLegacyManager.
    """
    contents = await file.read()
    metadata = {"filename": file.filename, "content_type": file.content_type}
    doc_id = await manager.ingest_legal_document(user_id, contents, metadata)
    return {"document_id": doc_id}

@router.post("/mezzo/dialogue/respond")
async def get_empathetic_response(
    request_data: Dict[str, Any],
    dialogue_engine: GriefDialogueEngine = Depends(get_grief_dialogue_engine_dependency)
):
    """
    Generates an empathetic response to user input using the GriefDialogueEngine.
    """
    user_input = request_data.get("user_input")
    history = request_data.get("history", [])
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is required.")
    
    response, sentiment = await dialogue_engine.generate_empathetic_response(user_input, history)
    return {"response": response, "sentiment": sentiment}

