import uvicorn
import asyncio

import api, llm
from executor import batched_inference_loop



def main():
    loop = asyncio.new_event_loop()

    loop.create_task(batched_inference_loop.run())
    loop.create_task(llm.run_llm_loop(loop))

    config = uvicorn.Config(app=api.app, loop=loop, host="0.0.0.0", port=30091)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
        

if __name__ == "__main__":
    main()