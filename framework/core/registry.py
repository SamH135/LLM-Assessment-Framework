# framework/core/registry.py
from typing import Dict, Type, List
import importlib
import pkgutil
import inspect
from pathlib import Path
from framework.core.base import BaseEvaluator, BaseLLMInterface


class Registry:
    """Central registry for evaluators and LLM interfaces"""

    _evaluators: Dict[str, Type[BaseEvaluator]] = {}
    _llm_interfaces: Dict[str, Type[BaseLLMInterface]] = {}

    @classmethod
    def discover_evaluators(cls) -> None:
        """Automatically discover and register all evaluators in the evaluators directory"""
        evaluators_path = Path(__file__).parent.parent / 'evaluators'
        failed_evaluators = []

        # Get all subdirectories in the evaluators directory
        for module_info in pkgutil.iter_modules([str(evaluators_path)]):
            if not module_info.ispkg:  # Skip non-package items
                continue

            # Import the evaluator module
            module_name = f"framework.evaluators.{module_info.name}.evaluator"
            try:
                module = importlib.import_module(module_name)

                # Find all classes in the module that inherit from BaseEvaluator
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                            issubclass(obj, BaseEvaluator) and
                            obj != BaseEvaluator):
                        try:
                            # Try to create an instance to verify initialization works
                            evaluator_instance = obj()
                            # If initialization succeeds, register the evaluator
                            cls.register_evaluator(obj)
                            print(f"Successfully registered evaluator: {obj.get_metadata().name}")
                        except Exception as init_error:
                            error_msg = f"Failed to initialize evaluator {name}: {str(init_error)}"
                            print(f"Warning: {error_msg}")
                            failed_evaluators.append({
                                "name": name,
                                "error": str(init_error)
                            })

            except Exception as e:
                error_msg = f"Failed to load evaluator module {module_info.name}: {str(e)}"
                print(f"Warning: {error_msg}")
                failed_evaluators.append({
                    "name": module_info.name,
                    "error": str(e)
                })

        if failed_evaluators:
            print("\nFailed evaluators summary:")
            for failure in failed_evaluators:
                print(f"- {failure['name']}: {failure['error']}")
    @classmethod
    def register_evaluator(cls, evaluator_class: Type[BaseEvaluator]) -> None:
        """Register an evaluator class"""
        metadata = evaluator_class.get_metadata()
        cls._evaluators[metadata.name] = evaluator_class

    @classmethod
    def register_llm(cls, llm_class: Type[BaseLLMInterface]) -> None:
        """Register an LLM interface class"""
        cls._llm_interfaces[llm_class.get_name()] = llm_class

    @classmethod
    def get_evaluator(cls, name: str) -> BaseEvaluator:
        """Get an instance of a registered evaluator"""
        if name not in cls._evaluators:
            raise ValueError(f"Unknown evaluator: {name}")
        return cls._evaluators[name]()

    @classmethod
    def get_llm(cls, name: str) -> BaseLLMInterface:
        """Get an instance of a registered LLM interface"""
        if name not in cls._llm_interfaces:
            raise ValueError(f"Unknown LLM interface: {name}")
        return cls._llm_interfaces[name]()

    @classmethod
    def list_evaluators(cls) -> List[str]:
        """Get list of registered evaluators"""
        return list(cls._evaluators.keys())

    @classmethod
    def list_llms(cls) -> List[str]:
        """Get list of registered LLM interfaces"""
        return list(cls._llm_interfaces.keys())

    @classmethod
    def discover_llm_interfaces(cls) -> None:
        """Automatically discover and register all LLM interfaces"""
        interfaces_path = Path(__file__).parent.parent / 'interfaces' / 'llm'

        for module_info in pkgutil.iter_modules([str(interfaces_path)]):
            if module_info.name == '__init__':  # Skip __init__.py
                continue

            module_name = f"framework.interfaces.llm.{module_info.name}"
            try:
                module = importlib.import_module(module_name)

                # Find all classes that inherit from BaseLLMInterface
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                            issubclass(obj, BaseLLMInterface) and
                            obj != BaseLLMInterface):
                        cls.register_llm(obj)
                        print(f"Registered LLM interface: {obj.get_name()}")

            except Exception as e:
                print(f"Warning: Failed to load LLM interface {module_info.name}: {e}")
