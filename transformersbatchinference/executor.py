from abc import ABC, abstractmethod
import asyncio
import time
from typing import Callable, Dict, List
import numpy as np
from torch.utils.data import Dataset

from models import BatchedCompletionRequest, CompletionRequest

from typing import Tuple


class BatchedDataset(Dataset):
    def __init__(self, data: List[BatchedCompletionRequest]):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i].prompt


class ExecutorQueue:
    datasets = []

    def __init__(self, fetch_new_data: Callable[[], List[BatchedDataset]]):
        self.fetch_new_data = fetch_new_data

    def put(self, dataset: BatchedDataset):
        self.datasets.append(dataset)
    
    async def pop(self):
        while self.is_empty():
            next_batch = self.fetch_new_data()
            if next_batch:
                self.put(next_batch)
            await asyncio.sleep(0.1)

        return self.datasets.pop(0)

    def is_empty(self):
        return len(self.datasets) == 0

class Synchronized(ABC):
    
    @abstractmethod
    def synchronize(self):
        pass

class BatchedDatasetBuffer(Synchronized):
    def __init__(self, queue: ExecutorQueue, max_size: int, max_ms_wait_time: int):
        self.max_size = max_size
        self.max_wait_time = max_ms_wait_time
        self.queue = queue
        self.buffer: List[BatchedCompletionRequest] = []
        self._reset_time()

    def _reset_time(self):
        self.time = time.time()

    def synchronize(self):
        if self._should_empty_buffer():
            dataset = self.empty_buffer()
            self.queue.put(dataset)
            
    def empty_buffer(self) -> BatchedDataset:
        dataset = BatchedDataset(self.buffer[:self.max_size])
        self.buffer = self.buffer[self.max_size-1:]
        self._reset_time()
        return dataset

    @property
    def size(self):
        return len(self.buffer)

    def is_full(self):
        return len(self.buffer) >= self.max_size
    
    def _should_empty_buffer(self):
        return self.is_full() or self.__is_time_threshold_reached()

    def __is_time_threshold_reached(self):
        if self.time + self.max_wait_time < time.time():
            self._reset_time()
            return not self.is_empty()
        
        return False
    
class BatchedInferenceLoop:

    def __init__(self, max_batch_size: int = 10, max_ms_wait_time: int = 1000):
        self.buffers: Dict[str, BatchedDatasetBuffer] = {}
        self.max_batch_size = max_batch_size
        self.max_ms_wait_time = max_ms_wait_time
    
    def add(self, request: CompletionRequest, request_id: str) -> BatchedCompletionRequest:
        key = request.key

        if key not in self.buffers:
            self.buffers[key] = BatchedDatasetBuffer(ExecutorQueue(self.fetch_new_data), self.max_batch_size, self.max_ms_wait_time)

        batched_request = BatchedCompletionRequest(**request.model_dump(), request_id=request_id)

        self.buffers[key].buffer.append(batched_request)
        return batched_request
    
    def fetch_new_data(self) -> BatchedDataset:
        if not self.buffers:
            return BatchedDataset([])
        
        index = np.argmax([queue.size for queue in self.buffers.values()])
        key = list(self.buffers.keys())[index]

        return self.buffers[key].empty_buffer()

    async def run(self):
        while True:
            for _, buffer in self.buffers.items():
                buffer.synchronize()

            await asyncio.sleep(0.1)


def initialize_loop() -> Tuple[BatchedInferenceLoop, ExecutorQueue]:
    loop = BatchedInferenceLoop()
    executor_queue = ExecutorQueue(loop.fetch_new_data)

    return loop, executor_queue

batched_inference_loop, executor_queue = initialize_loop()