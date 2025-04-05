# framework/core/test_case.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import uuid


class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TestPrompt:
    text: str
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    model_id: str
    model_name: str
    model_config: Dict[str, Any]
    prompts: List[TestPrompt]
    evaluator_ids: List[str]

    # Automatically generated fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TestStatus = TestStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def validate(self) -> bool:
        """Validate that the test case is properly configured"""
        if not self.model_id or not self.model_name:
            return False
        if not self.prompts:
            return False
        if not self.evaluator_ids:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_config": self.model_config,
            "prompts": [{"text": p.text, "category": p.category, "metadata": p.metadata} for p in self.prompts],
            "evaluator_ids": self.evaluator_ids,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": self.results,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        """Create from dictionary representation"""
        prompts = [TestPrompt(**p) for p in data["prompts"]]

        test_case = cls(
            model_id=data["model_id"],
            model_name=data["model_name"],
            model_config=data["model_config"],
            prompts=prompts,
            evaluator_ids=data["evaluator_ids"]
        )

        # Set the derived fields
        test_case.id = data["id"]
        test_case.status = TestStatus(data["status"])
        test_case.created_at = datetime.fromisoformat(data["created_at"])
        if data["started_at"]:
            test_case.started_at = datetime.fromisoformat(data["started_at"])
        if data["completed_at"]:
            test_case.completed_at = datetime.fromisoformat(data["completed_at"])
        test_case.results = data["results"]
        test_case.error = data["error"]

        return test_case
