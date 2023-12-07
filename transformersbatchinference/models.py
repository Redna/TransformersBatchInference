from typing import Dict, List, Optional
from pydantic import BaseModel

class CompletionRequest(BaseModel):
    inputs: str
    parameters: Optional[Dict] = None
    
    @property
    def key(self):
        return '-'.join([key for key in self.parameters.keys()])

class BatchedCompletionRequest(CompletionRequest):
    request_id: str
    generated_text: List[str] = None
    prompt_tokens: List[int] = [-1]
    new_tokens: List[int] = [-1]