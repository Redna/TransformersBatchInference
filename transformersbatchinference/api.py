import asyncio
import uuid
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from loguru import logger

# add cors middleware to fastapi app
from fastapi.middleware.cors import CORSMiddleware
from executor import batched_inference_loop

from models import Completion, CompletionRequest

app = FastAPI()

origins = ["http://localhost"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/v1/completion", response_model=Completion)
async def generate_completion(request: CompletionRequest):
    request_id = str(uuid.uuid4())

    logger.info(f"Batching completion for prompt: {request.prompt}")

    batched_request = batched_inference_loop.add(request, request_id=request_id)

    while not batched_request.completion:
        logger.info(f"Waiting for completion with id: {request_id}")
        await asyncio.sleep(0.5)

    return {"number_of_tokens": len(batched_request.completion), "completion": batched_request.completion}
