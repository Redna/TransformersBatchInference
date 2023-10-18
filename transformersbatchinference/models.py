from typing import Optional
from pydantic import BaseModel

class Completion(BaseModel):
    number_of_tokens: int
    completion: str

class CompletionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature:float =0.3
    top_p: float =0.95
    top_k: int =55
    repetition_penalty:float=1.15
    do_sample: bool = True

    @property
    def pipeline_kwargs(self):
        return dict(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty, 
            do_sample=self.do_sample)
    
    @property
    def key(self):
        return '-'.join([key for key in self.pipeline_kwargs.keys()])

class BatchedCompletionRequest(CompletionRequest):
    request_id: str
    completion: Optional[Completion] = None