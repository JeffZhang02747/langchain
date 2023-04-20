from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import poe

class PoeLLM(LLM):
    api_key: Any
    model: Any
    client: Any

    def __init__(self, api_key, model, **params):
        super().__init__(**params)
        self.api_key = api_key
        self.model = model
        self.client = poe.Client(api_key)

    @property
    def _llm_type(self) -> str:
        return "poe"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        for chunk in self.client.send_message(self.model, prompt, timeout=60):
            pass
        return chunk["text"]
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"api_key": self.api_key, "model": self.model}