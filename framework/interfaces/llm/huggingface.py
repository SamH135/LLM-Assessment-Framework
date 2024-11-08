# framework/interfaces/llm/huggingface.py
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ...core.base import BaseLLMInterface


class HuggingFaceLLM(BaseLLMInterface):
    """
    Interface for HuggingFace models.

    This class provides a flexible interface for various HuggingFace models.
    You can use any model from the HuggingFace Hub by specifying its name.

    Examples:
        # Using default model (gpt2)
        model = HuggingFaceLLM()

        # Using a specific model
        model = HuggingFaceLLM(
            model_name="facebook/opt-350m",
            task="text-generation",
            pipeline_kwargs={"device": "cpu"}
        )

        # Using a different type of model
        model = HuggingFaceLLM(
            model_name="google/flan-t5-base",
            task="text2text-generation"
        )

    Common model options:
        - Text Generation: "gpt2", "facebook/opt-350m", "EleutherAI/gpt-neo-125M"
        - Text2Text: "google/flan-t5-base", "facebook/bart-base"
        - Summarization: "facebook/bart-large-cnn", "google/pegasus-xsum"
    """

    def __init__(self,
                 model_name: str = "gpt2",  # Default model, but can be overridden
                 task: str = "text-generation",
                 model_kwargs: Dict[str, Any] = None,
                 pipeline_kwargs: Dict[str, Any] = None):
        """
        Initialize HuggingFace model interface.

        Args:
            model_name: Name of the model from HuggingFace Hub. Can be any valid model
                      identifier from huggingface.co/models
            task: Task type (e.g., "text-generation", "text2text-generation", "summarization")
            model_kwargs: Additional arguments for model initialization
            pipeline_kwargs: Additional arguments for pipeline configuration

        Raises:
            RuntimeError: If model initialization fails
            ValueError: If task type is not supported
        """
        self.model_name = model_name
        self.task = task
        self.model_kwargs = model_kwargs or {}
        self.pipeline_kwargs = pipeline_kwargs or {}

        try:
            self.generator = pipeline(
                task=self.task,
                model=self.model_name,
                **self.pipeline_kwargs
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize model {model_name}. "
                f"Error: {str(e)}. "
                f"Make sure the model name is correct and the model is compatible with the task."
            )

    # ... rest of the class implementation stays the same ...
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
