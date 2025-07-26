# backend/mezzo_anima_line/mezzo_api.py

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Dict, Any, List

from backend.dependencies import get_digital_legacy_manager, get_grief_dialogue_engine
from backend.mezzo_anima_line.digital_legacy_manager import DigitalLegacyManager
from backend.mezzo_anima_line.grief_dialogue_engine import GriefDialogueEngine

router = APIRouter()

@router.post("/legacy/document", status_code=201)
async def upload_legal_document(
    user_id: str,
    file: UploadFile = File(...),
    manager: DigitalLegacyManager = Depends(get_digital_legacy_manager)
):
    """
    Uploads a legal document for secure storage.
    """
    contents = await file.read()
    metadata = {"filename": file.filename, "content_type": file.content_type}
    doc_id = await manager.ingest_legal_document(user_id, contents, metadata)
    return {"document_id": doc_id}

@router.post("/dialogue/respond")
async def get_empathetic_response(
    request_data: Dict[str, Any],
    dialogue_engine: GriefDialogueEngine = Depends(get_grief_dialogue_engine)
):
    """
    Generates an empathetic response to user input.
    """
    user_input = request_data.get("user_input")
    history = request_data.get("history", [])
    if not user_input:
        raise HTTPException(status_code=400, detail="User input is required.")
    
    response, sentiment = await dialogue_engine.generate_empathetic_response(user_input, history)
    return {"response": response, "sentiment": sentiment}
