import asyncio
from typing import Dict, List
import uuid
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from loguru import logger
import time

# add cors middleware to fastapi app
from fastapi.middleware.cors import CORSMiddleware
from executor import batched_inference_loop

from models import BatchedCompletionRequest, CompletionRequest

simple_cache = dict()

app = FastAPI()

origins = ["http://localhost"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/v1/generation", response_model=BatchedCompletionRequest)
async def generate_completions(request: CompletionRequest):
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    cache_key = None
    if "cache_key" in request.parameters:
        cache_key = request.parameters["cache_key"]
        del request.parameters["cache_key"]

        if cache_key in simple_cache:
            logger.info(f"Found cache hit for request: {request_id}")
            return simple_cache[cache_key]
        
    
    logger.info(f"Batching request: {request_id}")

    batched_request = batched_inference_loop.add(request, request_id=request_id)

    logger.info(f"Waiting for completion with id: {request_id}")
    while not batched_request.generated_text:
        await asyncio.sleep(0.1)
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    logger.info(f"Completed request: {request_id} in {execution_time} seconds.")

    for i, (new_tokens, prompt_tokens) in enumerate(zip(batched_request.new_tokens, batched_request.prompt_tokens)):
        logger.info(f"Sequence {i} - prompt tokens: {prompt_tokens}, generated tokens: {new_tokens}")
    
    if cache_key:
        simple_cache[cache_key] = batched_request
    return batched_request
