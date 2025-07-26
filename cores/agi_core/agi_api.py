# backend/agi_core/agi_api.py

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from backend.dependencies import get_agi_core_manager
from backend.agi_core.agi_core_manager import AgiCoreManager

router = APIRouter()

@router.post("/task", response_model=Dict[str, Any])
async def run_agi_task(
    task_data: Dict[str, Any],
    agi_manager: AgiCoreManager = Depends(get_agi_core_manager)
):
    """
    Endpoint to trigger and manage AGI tasks.
    """
    task_type = task_data.get("task_type")
    params = task_data.get("params", {})

    if not task_type:
        raise HTTPException(status_code=400, detail="AGI task type not specified.")

    try:
        result = await agi_manager.run_agi_task(task_type, params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
