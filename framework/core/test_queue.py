# framework/core/test_queue.py
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from .test_case import TestCase, TestStatus


class TestQueueManager:
    """Manages test cases, including persistence"""

    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path) if storage_path else Path("./test_cases")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_test_cases()

    def _load_test_cases(self):
        """Load test cases from disk"""
        self.test_cases = {}
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    test_case = TestCase.from_dict(data)
                    self.test_cases[test_case.id] = test_case
            except Exception as e:
                print(f"Error loading test case {file_path}: {e}")

    def save_test_case(self, test_case: TestCase):
        """Save a test case to disk"""
        self.test_cases[test_case.id] = test_case

        file_path = self.storage_path / f"{test_case.id}.json"
        with open(file_path, "w") as f:
            json.dump(test_case.to_dict(), f, indent=2)

    def get_test_case(self, test_id: str) -> Optional[TestCase]:
        """Get a test case by ID"""
        return self.test_cases.get(test_id)

    def list_test_cases(self, status: Optional[TestStatus] = None) -> List[TestCase]:
        """List test cases, optionally filtered by status"""
        if status is None:
            return list(self.test_cases.values())
        return [tc for tc in self.test_cases.values() if tc.status == status]

    def delete_test_case(self, test_id: str) -> bool:
        """Delete a test case"""
        if test_id not in self.test_cases:
            return False

        # Remove from memory
        del self.test_cases[test_id]

        # Remove from disk
        file_path = self.storage_path / f"{test_id}.json"
        if file_path.exists():
            file_path.unlink()

        return True
