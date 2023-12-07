import asyncio
import torch

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from executor import executor_queue, BatchedDataset

model_checkpoint = "HuggingFaceH4/zephyr-7b-beta" #"mistralai/Mistral-7B-Instruct-v0.1" #

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto", torch_dtype=torch.float16)
#tokenizer.pad_token_id = model.config.eos_token_id
#tokenizer.padding_side = "left"
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

async def run_llm_loop(event_loop: asyncio.AbstractEventLoop):
    while True:
        batch: BatchedDataset = await executor_queue.pop()

        def _run_pipeline(batch):
            try:
                completions = [completion for completion in generator(batch, batch_size=len(batch), **batch.data[0].parameters)]
                torch.cuda.empty_cache()
                return completions
            except Exception as e:
                logger.error(e)
                return [{"error": e} for _ in range(len(batch))]

        completions = await event_loop.run_in_executor(None, _run_pipeline, batch)
        
        for request, completion in zip(batch.data, completions):
            request.generated_text = completion
            request.prompt_tokens = [len(tokenizer.encode(request.inputs, add_special_tokens=False))] * len(completion)
            request.new_tokens = [len(tokenizer.encode(c["generated_text"], add_special_tokens=False)) - request.prompt_tokens[0] for c in completion]

        logger.info("Waiting for the next batch...")
        await asyncio.sleep(0.01)