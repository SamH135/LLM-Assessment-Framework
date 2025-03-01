# test_api.py
import requests
import json
from pathlib import Path
import time
from typing import Any, Dict
import sys

BASE_URL = "http://localhost:8000/api"


class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.response = None
        self.duration = 0


class APITester:
    def __init__(self):
        self.results = []
        self.session = requests.Session()

    def print_header(self, message: str) -> None:
        print(f"\n{'=' * 80}")
        print(f" {message}")
        print(f"{'=' * 80}")

    def print_subheader(self, message: str) -> None:
        print(f"\n{'-' * 40}")
        print(f" {message}")
        print(f"{'-' * 40}")

    def print_success(self, message: str) -> None:
        print(f"✓ {message}")

    def print_error(self, message: str) -> None:
        print(f"✗ {message}")

    def print_info(self, message: str) -> None:
        print(f"ℹ {message}")

    def validate_response_structure(self, response_data: Dict[str, Any]) -> bool:
        """Validate basic response structure"""
        return isinstance(response_data, dict) and "success" in response_data and "data" in response_data

    def run_test(self, name: str, test_func) -> TestResult:
        """Run a test and record its result"""
        result = TestResult(name)
        start_time = time.time()

        try:
            test_func(result)
            result.passed = True
        except Exception as e:
            result.error = str(e)
        finally:
            result.duration = time.time() - start_time
            self.results.append(result)

        return result

    def test_evaluators_endpoint(self, result: TestResult) -> None:
        """Test GET /api/evaluators"""
        self.print_subheader("Testing GET /api/evaluators")

        response = self.session.get(f"{BASE_URL}/evaluators")
        result.response = response

        if response.status_code != 200:
            raise Exception(f"Expected status code 200, got {response.status_code}")

        data = response.json()
        if not self.validate_response_structure(data):
            raise Exception("Invalid response structure")

        evaluators = data["data"]
        if not isinstance(evaluators, list):
            raise Exception("Evaluators data should be a list")

        # Validate each evaluator has complete metadata
        required_fields = {"id", "name", "description", "version", "category", "tags"}
        for evaluator in evaluators:
            missing_fields = required_fields - set(evaluator.keys())
            if missing_fields:
                raise Exception(
                    f"Missing required fields in evaluator {evaluator.get('name', 'Unknown')}: {missing_fields}")

            # Validate required field values
            if not evaluator.get('description'):
                raise Exception(f"Missing description for evaluator: {evaluator.get('name', 'Unknown')}")
            if not evaluator.get('version'):
                raise Exception(f"Missing version for evaluator: {evaluator.get('name', 'Unknown')}")

            if not isinstance(evaluator["tags"], list):
                raise Exception(f"Tags should be a list for evaluator: {evaluator.get('name', 'Unknown')}")

        self.print_success(f"Found {len(evaluators)} evaluators:")
        for evaluator in evaluators:
            self.print_info(f"- {evaluator['name']} (v{evaluator['version']})")
            print(f"  Description: {evaluator['description']}")
            print(f"  Category: {evaluator['category']}")
            print(f"  Tags: {', '.join(evaluator['tags'])}")
        print()

    def test_models_endpoint(self, result: TestResult) -> None:
        """Test GET /api/models"""
        self.print_subheader("Testing GET /api/models")

        response = self.session.get(f"{BASE_URL}/models")
        result.response = response

        if response.status_code != 200:
            raise Exception(f"Expected status code 200, got {response.status_code}")

        data = response.json()
        if not self.validate_response_structure(data):
            raise Exception("Invalid response structure")

        models = data["data"]
        if not isinstance(models, list):
            raise Exception("Models data should be a list")

        required_fields = {"id", "name", "configuration_options"}
        for model in models:
            missing_fields = required_fields - set(model.keys())
            if missing_fields:
                raise Exception(f"Missing required fields in model: {missing_fields}")

            if not isinstance(model["configuration_options"], dict):
                raise Exception("Model configuration_options should be a dictionary")

        self.print_success(f"Found {len(models)} models:")
        for model in models:
            self.print_info(f"- {model['name']}")
            if model["configuration_options"]:
                print("  Configuration options:")
                for option, details in model["configuration_options"].items():
                    print(f"    - {option}: {details.get('description', 'No description')}")
        print()

    def test_prompts_loading(self, result: TestResult) -> None:
        """Test POST /api/prompts/load"""
        self.print_subheader("Testing POST /api/prompts/load")

        example_path = Path("framework/examples/prompts/agencyPrompts.txt")
        if not example_path.exists():
            raise Exception(f"Example prompts file not found: {example_path}")

        payload = {"file_path": str(example_path)}

        self.print_info("Sending request with file path:")
        print(f"  {example_path}\n")

        response = self.session.post(f"{BASE_URL}/prompts/load", json=payload)
        result.response = response

        if response.status_code != 200:
            raise Exception(f"Expected status code 200, got {response.status_code}")

        data = response.json()
        if not self.validate_response_structure(data):
            raise Exception("Invalid response structure")

        prompts_data = data["data"]
        if not isinstance(prompts_data, dict) or "categories" not in prompts_data or "prompts" not in prompts_data:
            raise Exception("Invalid prompts data structure")

        categories = prompts_data["categories"]
        prompts = prompts_data["prompts"]

        if not categories:
            raise Exception("No categories found in response")

        self.print_success("Successfully loaded prompts file")
        self.print_info(f"Found {len(categories)} categories:")

        # Print each category with proper formatting and error checking
        for category in categories:
            if category not in prompts:
                raise Exception(f"Category {category} has no prompts data")
            category_prompts = prompts[category]
            self.print_info(f"  - {category}: {len(category_prompts)} prompts")

        # Add final newline for spacing
        print()

    def test_run_test_endpoint(self, result: TestResult) -> None:
        """Test POST /api/test"""
        self.print_subheader("Testing POST /api/test")

        test_data = {
            "model_type": "HuggingFace Model",
            "configuration": {
                "model_name": "gpt2",
                "max_length": 50
            },
            "prompt": "What is the meaning of life?",
            "selected_tests": ["Agency Analysis"]
        }

        self.print_info("Sending test request with data:")
        print(json.dumps(test_data, indent=2))
        print()

        response = self.session.post(f"{BASE_URL}/test", json=test_data)
        result.response = response

        if response.status_code != 200:
            raise Exception(f"Expected status code 200, got {response.status_code}")

        data = response.json()
        if not self.validate_response_structure(data):
            raise Exception("Invalid response structure")

        test_results = data["data"]
        required_fields = {"prompt", "response", "results", "metadata"}
        missing_fields = required_fields - set(test_results.keys())
        if missing_fields:
            raise Exception(f"Missing required fields in test results: {missing_fields}")

        self.print_success("Test completed successfully")
        print()

        self.print_info("Model Response:")
        print(f"  {test_results['response'][:100]}...")
        print()

        for test_name, test_result in test_results["results"].items():
            self.print_info(f"Results for {test_name}:")
            if "summary" in test_result:
                print(f"  Score: {test_result['summary']['score']}")
                print(f"  Risk Level: {test_result['summary']['risk_level']}")
                print("  Key Findings:")
                for finding in test_result['summary']['key_findings']:
                    print(f"    - {finding}")
            print()

    def run_all_tests(self) -> None:
        """Run all API tests"""
        self.print_header("Running API Tests")

        tests = [
            ("Evaluators Endpoint", self.test_evaluators_endpoint),
            ("Models Endpoint", self.test_models_endpoint),
            ("Prompts Loading", self.test_prompts_loading),
            ("Test Running", self.test_run_test_endpoint)
        ]

        for test_name, test_func in tests:
            try:
                result = self.run_test(test_name, test_func)
                if result.passed:
                    self.print_success(f"{test_name} - Passed ({result.duration:.2f}s)")
                else:
                    self.print_error(f"{test_name} - Failed: {result.error}")
            except KeyboardInterrupt:
                self.print_error("\nTesting interrupted by user")
                break
            except Exception as e:
                self.print_error(f"Unexpected error in {test_name}: {str(e)}")

        self.print_summary()

    def print_summary(self) -> None:
        """Print test results summary"""
        self.print_header("Test Summary")

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total Duration: {sum(r.duration for r in self.results):.2f}s")

        if failed > 0:
            self.print_header("Failed Tests")
            for result in self.results:
                if not result.passed:
                    self.print_error(f"{result.name}: {result.error}")


def main():
    try:
        tester = APITester()
        tester.run_all_tests()
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the API server at {BASE_URL}")
        print("Make sure the server is running and try again.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()