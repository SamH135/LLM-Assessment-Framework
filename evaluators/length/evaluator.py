# framework/evaluators/length/evaluator.py
from typing import Dict, Any, List
from framework.core.base import BaseEvaluator, EvaluatorMetadata


class ResponseLengthEvaluator(BaseEvaluator):
    """Evaluator for checking if responses are within acceptable length bounds."""

    def __init__(self):
        self.min_length = 20  # minimum acceptable words
        self.max_length = 500  # maximum acceptable words

    @classmethod
    def get_metadata(cls) -> EvaluatorMetadata:
        return EvaluatorMetadata(
            name="Response Length Analysis",
            description="Evaluates if responses are within acceptable length bounds",
            version="1.0.0",
            category="Quality",
            tags=["length", "conciseness", "verbosity"]
        )

    def evaluate(self, text: str) -> Dict[str, Any]:
        """Evaluate the length characteristics of the response."""
        # Split into words and count
        words = text.split()
        word_count = len(words)

        # Calculate scores and metrics
        too_short = word_count < self.min_length
        too_long = word_count > self.max_length
        ideal_range = not (too_short or too_long)

        # Calculate a basic score (100 for ideal, less for too short/long)
        if ideal_range:
            score = 100
        else:
            if too_short:
                score = (word_count / self.min_length) * 100
            else:
                score = max(0, 100 - ((word_count - self.max_length) / self.max_length * 100))

        return {
            'word_count': word_count,
            'too_short': too_short,
            'too_long': too_long,
            'ideal_range': ideal_range,
            'score': score,
            'min_length': self.min_length,
            'max_length': self.max_length
        }

    def interpret(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable interpretation of the length evaluation results."""
        interpretation = []

        # Core length assessment
        interpretation.append(f"Word Count: {results['word_count']}")
        interpretation.append(f"Score: {results['score']:.2f}")

        if results['ideal_range']:
            interpretation.append("PASS: Response length is within acceptable bounds.")
        else:
            if results['too_short']:
                interpretation.append(
                    f"ALERT: Response is too short. Minimum recommended length is "
                    f"{results['min_length']} words.")
            if results['too_long']:
                interpretation.append(
                    f"ALERT: Response is too long. Maximum recommended length is "
                    f"{results['max_length']} words.")

        return " ".join(interpretation)

    def summarize_category_results(self, category_results: List[Dict[str, Any]]) -> str:
        """Summarize length evaluation results for a category."""
        if not category_results:
            return "No results to summarize"

        total_prompts = len(category_results)
        total_score = 0
        too_short_count = 0
        too_long_count = 0
        ideal_count = 0
        total_words = 0

        for result in category_results:
            if 'results' in result['tests'].get('Response Length Analysis', {}):
                test_result = result['tests']['Response Length Analysis']['results']
                total_score += test_result.get('score', 0)
                total_words += test_result.get('word_count', 0)

                if test_result.get('too_short', False):
                    too_short_count += 1
                elif test_result.get('too_long', False):
                    too_long_count += 1
                else:
                    ideal_count += 1

        avg_score = total_score / total_prompts if total_prompts > 0 else 0
        avg_words = total_words / total_prompts if total_prompts > 0 else 0

        summary = [
            f"Total Prompts Tested: {total_prompts}",
            f"Average Score: {avg_score:.2f}",
            f"Average Word Count: {avg_words:.1f}",
            f"Length Distribution:",
            f"  Too Short: {too_short_count}",
            f"  Ideal Length: {ideal_count}",
            f"  Too Long: {too_long_count}"
        ]

        return "\n".join(summary)