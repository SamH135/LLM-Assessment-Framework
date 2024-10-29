# framework/interfaces/llm/huggingface.py
from typing import Optional
from transformers import pipeline
from ...core.base import BaseLLMInterface


class HuggingFaceLLM(BaseLLMInterface):
    """Interface for HuggingFace models"""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.generator = pipeline('text-generation', model=model_name)

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate response using HuggingFace model"""
        response = self.generator(prompt, max_length=max_length, num_return_sequences=1)[0]
        # Remove the prompt from the response if it's included
        generated_text = response['generated_text']
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        return generated_text

    @classmethod
    def get_name(cls) -> str:
        return "HuggingFace Model"