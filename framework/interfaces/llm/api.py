# framework/interfaces/llm/api.py
import requests
from ...core.base import BaseLLMInterface


class ApiLLM(BaseLLMInterface):
    """Interface for API-based LLMs"""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using API endpoint"""
        data = {
            "prompt": prompt,
            **kwargs
        }
        response = requests.post(self.api_url, headers=self.headers, json=data)
        response.raise_for_status()
        return response.json()["response"]

    @classmethod
    def get_name(cls) -> str:
        return "API Model"