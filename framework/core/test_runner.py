# framework/core/test_runner.py
from typing import Dict, Any, List, Optional, Callable
import asyncio
from datetime import datetime
from .test_case import TestCase, TestStatus
from .registry import Registry


class TestRunner:
    def __init__(self):
        self.active_tests = {}  # id -> task
        self.queue = []  # List of TestCase objects
        self.on_test_complete_callbacks = []

    def add_on_complete_callback(self, callback: Callable[[TestCase], None]):
        self.on_test_complete_callbacks.append(callback)

    def enqueue(self, test_case: TestCase) -> str:
        """Add a test case to the queue"""
        if not test_case.validate():
            raise ValueError("Invalid test case configuration")

        self.queue.append(test_case)
        return test_case.id

    def cancel(self, test_id: str) -> bool:
        """Cancel a test by ID"""
        # First check the queue
        for i, test in enumerate(self.queue):
            if test.id == test_id:
                test.status = TestStatus.CANCELLED
                self.queue.pop(i)
                return True

        # Then check active tests
        if test_id in self.active_tests:
            # Set status to cancelled
            self.active_tests[test_id].test_case.status = TestStatus.CANCELLED
            # Cancel the task
            self.active_tests[test_id].task.cancel()
            return True

        return False

    async def start(self, max_concurrent: int = 3):
        """Start processing the test queue"""
        while True:
            # Process queue if we have capacity
            while self.queue and len(self.active_tests) < max_concurrent:
                test_case = self.queue.pop(0)
                await self._run_test(test_case)

            # Wait a bit before checking again
            await asyncio.sleep(1)

    async def _run_test(self, test_case: TestCase):
        """Run a single test case"""
        test_case.status = TestStatus.RUNNING
        test_case.started_at = datetime.now()

        # Create task for the test
        task = asyncio.create_task(self._execute_test(test_case))
        self.active_tests[test_case.id] = {"test_case": test_case, "task": task}

        # Set up completion handler
        task.add_done_callback(lambda t: self._handle_test_completion(test_case.id, t))

    async def _execute_test(self, test_case: TestCase):
        """Execute the actual test"""
        try:
            # Get the model interface
            llm = Registry.get_llm(test_case.model_id)
            for key, value in test_case.model_config.items():
                setattr(llm, key, value)

            # Initialize results
            results = []

            # Process each prompt
            for prompt in test_case.prompts:
                # Skip if the test was cancelled
                if test_case.status == TestStatus.CANCELLED:
                    break

                # Generate response
                response = llm.generate_response(prompt.text)

                # Run evaluators
                prompt_results = {
                    "prompt": prompt.text,
                    "category": prompt.category,
                    "response": response,
                    "evaluations": {}
                }

                for evaluator_id in test_case.evaluator_ids:
                    evaluator = Registry.get_evaluator(evaluator_id)
                    evaluation = evaluator.evaluate(response)
                    interpretation = evaluator.interpret(evaluation)

                    prompt_results["evaluations"][evaluator_id] = {
                        "results": evaluation,
                        "interpretation": interpretation
                    }

                results.append(prompt_results)

            # Update test case with results
            test_case.results = results
            test_case.status = TestStatus.COMPLETED

        except Exception as e:
            test_case.status = TestStatus.FAILED
            test_case.error = str(e)

        finally:
            test_case.completed_at = datetime.now()
            return test_case

    def _handle_test_completion(self, test_id: str, task):
        """Handle test completion"""
        if test_id in self.active_tests:
            # Remove from active tests
            del self.active_tests[test_id]

        # Don't proceed if cancelled
        if task.cancelled():
            return

        try:
            # Get the completed test case
            test_case = task.result()

            # Notify callbacks
            for callback in self.on_test_complete_callbacks:
                callback(test_case)

        except Exception as e:
            print(f"Error handling test completion: {e}")
