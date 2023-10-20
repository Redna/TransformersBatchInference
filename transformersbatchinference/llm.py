import asyncio
import torch

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from executor import executor_queue, BatchedDataset

model_checkpoint = "HuggingFaceH4/zephyr-7b-alpha"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto", torch_dtype=torch.bfloat16)


async def run_llm_loop(event_loop: asyncio.AbstractEventLoop):
    while True:
        batch: BatchedDataset = await executor_queue.pop()

        def _run_pipeline(batch):
            try:
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    **batch.data[0].parameters
                )
            
                return [completion for completion in pipe(batch, batch_size=len(batch))]
        
            except Exception as e:
                print(e)
                return [{"error": e} for _ in range(len(batch))]

        completions = await event_loop.run_in_executor(None, _run_pipeline, batch)
        
        for request, completion in zip(batch.data, completions):
            request.generated_text = completion

        logger("Waiting for the next batch...")
        await asyncio.sleep(0.01)