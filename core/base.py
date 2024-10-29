from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class EvaluatorMetadata:
    name: str
    description: str
    version: str
    category: str
    tags: List[str]


class BaseEvaluator(ABC):
    @classmethod
    @abstractmethod
    def get_metadata(cls) -> EvaluatorMetadata:
        pass

    @abstractmethod
    def evaluate(self, text: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def interpret(self, results: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def summarize_category_results(self, category_results: List[Dict[str, Any]]) -> str:
        """
        Summarize results for a category of prompts.
        Args:
            category_results: List of results for prompts in a category
        Returns:
            A formatted string containing the summary
        """
        pass


class BaseLLMInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        pass
