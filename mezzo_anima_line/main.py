# backend/mezzo_anima_line/main.py

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional

# MEZZO-specific imports
from . import dependencies
from .mezzo_api import router as mezzo_router
from .config import settings  # Assuming MEZZO will have its own settings
from .emotional_sentinel import EmotionalSentinel

# Shared services from the ecosystem
from backend.communication.notification_manager import NotificationManager
from backend.support_ai.core_ai_handler import LLMClient

# Firebase/Firestore
from firebase_admin import initialize_app, firestore

# Google Cloud Logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler, setup_logging

# --- Centralized Logging Configuration ---
client = google.cloud.logging.Client()
handler = CloudLoggingHandler(client)
setup_logging(handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("uvicorn.access").addHandler(handler)
logging.getLogger("uvicorn.error").addHandler(handler)

logger.info("Logging configured for MEZZO Anima Line Service.")

# --- Lifespan Management for MEZZO Service ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- MEZZO Anima Line Service Startup ---")
    
    # Initialize Firebase Admin SDK for MEZZO
    try:
        if not settings.GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID is not set for MEZZO.")
        
        initialize_app(options={'projectId': settings.GCP_PROJECT_ID}, name='mezzoApp')
        db = firestore.client()
        dependencies.set_firestore_db(db)
        logger.info("Firebase/Firestore for MEZZO initialized successfully.")
    except Exception as e:
        logger.critical(f"FATAL: Failed to initialize Firebase/Firestore for MEZZO: {e}", exc_info=True)
        # In a real scenario, you might want the app to fail startup if DB connection fails
        
    # Initialize all MEZZO-specific singletons
    dependencies.initialize_mezzo_singletons()
    logger.info("All MEZZO singletons initialized.")

    yield
    
    logger.info("--- MEZZO Anima Line Service Shutdown ---")
    # Add any cleanup logic here if necessary

# --- FastAPI App Definition ---
app = FastAPI(
    title="MEZZO Anima Line API",
    description="A dedicated service for digital legacy and emotional companionship.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(mezzo_router, prefix="/api/v1", tags=["MEZZO Anima Line"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "MEZZO Anima Line Service is operational."}

if __name__ == "__main__":
    uvicorn.run("backend.mezzo_anima_line.main:app", host="0.0.0.0", port=8001, reload=True)
