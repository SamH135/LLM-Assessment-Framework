# framework/interfaces/llm/huggingface.py
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from ...core.base import BaseLLMInterface


class HuggingFaceLLM(BaseLLMInterface):
    """
    Interface for HuggingFace models.

    Examples:
        # Using GPT-2 (Causal LM)
        model = HuggingFaceLLM(model_name="gpt2", task="text-generation")

        # Using BART for Q&A
        model = HuggingFaceLLM(
            model_name="vblagoje/bart_lfqa",
            task="text2text-generation",
            prompt_template="Question: {input} Answer:"
        )

        # Using T5 for open-ended generation
        model = HuggingFaceLLM(
            model_name="google/flan-t5-base",
            task="text2text-generation",
            prompt_template="{input}"
        )
    """

    def __init__(self,
                 model_name: str = "gpt2",
                 task: str = "text-generation",
                 prompt_template: Optional[str] = None,
                 model_kwargs: Dict[str, Any] = None,
                 pipeline_kwargs: Dict[str, Any] = None):
        """
        Initialize HuggingFace model interface.

        Args:
            model_name: Name of the model from HuggingFace Hub
            task: Task type ("text-generation" or "text2text-generation")
            prompt_template: Template for formatting input. Use {input} as placeholder
            model_kwargs: Additional arguments for model initialization
            pipeline_kwargs: Additional arguments for pipeline configuration
        """
        self.model_name = model_name
        self.task = task
        self.prompt_template = prompt_template
        self.model_kwargs = model_kwargs or {}

        # Set default generation parameters based on model architecture
        default_params = self._get_default_params(model_name)
        self.pipeline_kwargs = {**default_params, **(pipeline_kwargs or {})}

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

    def _get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters based on model architecture."""
        if any(name in model_name.lower() for name in ["bart", "t5", "flan"]):
            return {
                "max_length": 128,
                "min_length": 20,
                "num_beams": 4,
                "temperature": 0.7,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            }
        else:  # Default parameters for GPT-like models
            return {
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }

    def _format_prompt(self, prompt: str) -> str:
        """Format the input prompt using the template if provided."""
        if self.prompt_template:
            return self.prompt_template.format(input=prompt)
        return prompt

    def generate_response(self, prompt: str, max_length: Optional[int] = None) -> str:
        """Generate response using HuggingFace model"""
        formatted_prompt = self._format_prompt(prompt)

        # Override max_length if provided
        if max_length:
            self.pipeline_kwargs["max_length"] = max_length

        try:
            if self.task == "text2text-generation":
                # Handle text2text models (BART, T5, etc.)
                response = self.generator(
                    formatted_prompt,
                    **self.pipeline_kwargs
                )
                return response[0]['generated_text'].strip()
            else:
                # Handle causal language models (GPT-2, OPT, etc.)
                response = self.generator(
                    formatted_prompt,
                    **self.pipeline_kwargs
                )[0]
                generated_text = response['generated_text']
                if generated_text.startswith(formatted_prompt):
                    generated_text = generated_text[len(formatted_prompt):].strip()
                return generated_text

        except Exception as e:
            return f"Error generating response: {str(e)}"

    @classmethod
    def get_name(cls) -> str:
        return "HuggingFace Model"
