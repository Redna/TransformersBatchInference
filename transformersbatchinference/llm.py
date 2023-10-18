import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from executor import executor_queue, BatchedDataset

model_checkpoint = "HuggingFaceH4/zephyr-7b-alpha"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto", torch_dtype=torch.bfloat16)


async def run_llm_loop(event_loop: asyncio.AbstractEventLoop):
    global model, tokenizer
    
    while True:
        batch: BatchedDataset = await executor_queue.pop()

        def _run_pipeline(batch):
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **batch.data[0].pipeline_kwargs
            )
            
            return [completion for completion in pipe(batch, batch_size=len(batch))]

        completions = await event_loop.run_in_executor(None, _run_pipeline, batch)
        
        for request, completion in zip(batch.data, completions):
            request.completion = completion[-1]["generated_text"]

        print("waiting for the next batch...")
        asyncio.sleep(0.1)

    