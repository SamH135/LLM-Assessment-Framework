# framework/main.py
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from framework.core.registry import Registry
from framework.utils.prompts import PromptCollection, load_prompts
from framework.core.base import BaseLLMInterface, BaseEvaluator


class TestingFramework:
    def __init__(self):
        """Initialize the framework and discover evaluators"""
        self.llm: Optional[BaseLLMInterface] = None
        self.prompts: Optional[PromptCollection] = None
        self.selected_categories: Optional[List[str]] = None
        self.selected_tests: Optional[List[str]] = None
        self.results_history: List[Dict[str, Any]] = []

        # Auto-discover evaluators
        Registry.discover_evaluators()

        # Register default LLM interfaces
        from .interfaces.llm.huggingface import HuggingFaceLLM
        from .interfaces.llm.api import ApiLLM
        Registry.register_llm(HuggingFaceLLM)
        Registry.register_llm(ApiLLM)

    def run_menu(self) -> None:
        """Display and handle the main menu."""
        while True:
            self._display_menu()
            choice = input("\nEnter your choice (1-6): ")

            options = {
                "1": self.select_llm,
                "2": self.select_prompts,
                "3": self.select_tests,
                "4": self.run_tests,
                "5": self.view_results_summary,
                "6": self.exit_framework
            }

            if choice in options:
                options[choice]()
                if choice == "6":
                    break
            else:
                print("Invalid choice. Please try again.")

    def _display_menu(self) -> None:
        """Display the main menu options."""
        print("\n=== LLM Testing Framework ===")
        print("1. Select LLM")
        print("2. Load Prompts")
        print("3. Select Tests")
        print("4. Run Tests")
        print("5. View Results Summary")
        print("6. Exit")

    def select_llm(self) -> None:
        """Handle LLM selection."""
        print("\n=== Select LLM ===")
        available_llms = Registry.list_llms()

        for i, llm in enumerate(available_llms, 1):
            print(f"{i}. {llm}")

        try:
            choice = int(input("\nEnter your choice: "))
            if 1 <= choice <= len(available_llms):
                self.llm = Registry.get_llm(available_llms[choice - 1])
                print(f"Selected: {available_llms[choice - 1]}")
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")

    def select_prompts(self) -> None:
        """Handle prompt file selection and loading."""
        print("\n=== Select Prompts ===")
        file_path = input("Enter path to prompts file: ")

        try:
            self.prompts = load_prompts(file_path)
            print(f"\nLoaded prompts with {len(self.prompts.get_categories())} categories:")

            for i, category in enumerate(self.prompts.get_categories(), 1):
                prompt_count = len(self.prompts.get_prompts([category]))
                print(f"{i}. {category} ({prompt_count} prompts)")

            self.select_categories()
        except FileNotFoundError:
            print("Error: File not found")
        except Exception as e:
            print(f"Error loading prompts: {e}")

    def select_categories(self) -> None:
        """Handle category selection for testing."""
        if not self.prompts:
            print("Error: No prompts loaded")
            return

        print("\n=== Select Categories ===")
        print("Enter category numbers to test (comma-separated)")
        print("Leave empty to test all categories")

        choice = input("\nCategories: ").strip()

        if not choice:
            self.selected_categories = None
            print("Selected all categories")
        else:
            try:
                indices = [int(c.strip()) - 1 for c in choice.split(",")]
                categories = self.prompts.get_categories()
                self.selected_categories = [categories[i] for i in indices]
                print(f"Selected categories: {', '.join(self.selected_categories)}")
            except (ValueError, IndexError):
                print("Invalid selection")
                self.selected_categories = None

    def select_tests(self) -> None:
        """Handle test selection."""
        print("\n=== Select Tests ===")
        available_tests = Registry.list_evaluators()

        for i, test in enumerate(available_tests, 1):
            evaluator = Registry.get_evaluator(test)
            metadata = evaluator.get_metadata()
            print(f"{i}. {metadata.name}: {metadata.description}")

        choice = input("\nEnter test numbers (comma-separated): ")
        try:
            indices = [int(c.strip()) - 1 for c in choice.split(",")]
            self.selected_tests = [available_tests[i] for i in indices]
            print(f"Selected tests: {', '.join(self.selected_tests)}")
        except (ValueError, IndexError):
            print("Invalid selection")
            self.selected_tests = []

    def run_tests(self) -> None:
        """Run selected tests on selected prompts."""
        if not self._validate_test_setup():
            return

        test_prompts = self.prompts.get_prompts(self.selected_categories)

        print("\n=== Running Tests ===")
        print(f"Testing {len(test_prompts)} prompts")

        current_results = []

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nPrompt {i} [{prompt.category}]:")
            print(f"Text: {prompt.text}")

            try:
                response = self.llm.generate_response(prompt.text)
                print(f"\nResponse: {response}\n")

                prompt_results = {
                    'prompt': prompt,
                    'response': response,
                    'tests': {}
                }

                for test_name in self.selected_tests:
                    print(f"Running {test_name}...")
                    evaluator = Registry.get_evaluator(test_name)
                    results = evaluator.evaluate(response)
                    interpretation = evaluator.interpret(results)

                    prompt_results['tests'][test_name] = {
                        'results': results,
                        'interpretation': interpretation
                    }

                    print(f"\nResults:\n{interpretation}\n")
                    print("-" * 80)

                current_results.append(prompt_results)

            except Exception as e:
                print(f"Error processing prompt: {e}")

        self.results_history.append({
            'timestamp': datetime.now(),
            'results': current_results
        })

    def _validate_test_setup(self) -> bool:
        """Validate that all necessary components are selected."""
        if not self.llm:
            print("Error: No LLM selected")
            return False

        if not self.prompts:
            print("Error: No prompts loaded")
            return False

        if not self.selected_tests:
            print("Error: No tests selected")
            return False

        return True

    def view_results_summary(self) -> None:
        """Display summary of test results."""
        if not self.results_history:
            print("No test results available")
            return

        print("\n=== Test Results Summary ===")
        for i, session in enumerate(self.results_history, 1):
            print(f"\nTest Session {i} - {session['timestamp']}")
            self._display_session_summary(session['results'])

    def _display_session_summary(self, results: List[Dict[str, Any]]) -> None:
        """Display summary for a single test session."""
        # Group results by category
        by_category = {}
        for result in results:
            category = result['prompt'].category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)

        # Display summary for each category
        for category, category_results in by_category.items():
            print(f"\nCategory: {category}")

            # Get summaries from each evaluator
            for test_name in self.selected_tests:
                print(f"\n{test_name} Results:")
                evaluator = Registry.get_evaluator(test_name)
                summary = evaluator.summarize_category_results(category_results)
                print(summary)

    def exit_framework(self) -> None:
        """Clean up and exit the framework."""
        print("\nThank you for using the LLM Testing Framework!")
        # Add any cleanup code here if needed


# Additional imports needed at the top:
