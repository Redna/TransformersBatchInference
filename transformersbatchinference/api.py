import asyncio
from typing import Dict, List
import uuid
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from loguru import logger

# add cors middleware to fastapi app
from fastapi.middleware.cors import CORSMiddleware
from executor import batched_inference_loop

from models import CompletionRequest

app = FastAPI()

origins = ["http://localhost"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/v1/generation", response_model=List[Dict])
async def generate_completions(request: CompletionRequest):
    request_id = str(uuid.uuid4())

    logger.info(f"Batching request: {request_id}")

    batched_request = batched_inference_loop.add(request, request_id=request_id)

    logger.info(f"Waiting for completion with id: {request_id}")
    while not batched_request.generated_text:
        await asyncio.sleep(0.1)

    logger.info(f"Completed request: {request_id}")
    return batched_request.generated_text
