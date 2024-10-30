# LLM Testing Framework

A framework for evaluating and testing Large Language Models (LLMs) across various dimensions including agency detection and response length analysis.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-testing-framework.git
cd llm-testing-framework
```

2. (Optional but recommended) Create and activate a virtual environment:
```bash
# Using venv
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install the framework and its dependencies:
```bash
pip install -e .
```

This will install all required dependencies, including:
- transformers (≥4.30.0)
- torch (≥2.0.0)
- numpy (≥1.24.0)
- tokenizers (≥0.13.0)
- tqdm (≥4.65.0)
- datasets (≥2.10.0)
- huggingface-hub (≥0.16.0)

## Project Structure
```
framework/
├── core/               # Core framework components
├── evaluators/         # Test evaluators
│   ├── agency/        # Agency detection
│   └── length/        # Response length analysis
├── examples/          # Example prompts and usage
├── interfaces/        # LLM interfaces
│   └── llm/          # Model interfaces
└── utils/            # Utility functions
```

## Usage

1. Run the framework (from the CLI/terminal - make sure to cd to the project directory):
```bash
python run.py
```

2. Follow the interactive menu to:
   - Select an LLM interface (HuggingFace models available by default)
   - Load test prompts
   - Select evaluation tests
   - Run evaluations
   - View results

### Example Prompt File Format
```
Category1
prompt1
prompt2

Category2
prompt3
prompt4
```

## Available Tests

### Agency Analysis
Evaluates the level of agency expressed in AI responses, detecting:
- High agency phrases
- Capability claims
- Disclaimers
- Action verbs
- Emotional expression

### Response Length Analysis
Analyzes response length characteristics including:
- Word count
- Character count
- Sentence structure

## Adding New Tests

1. Create a new directory under `framework/evaluators/`
2. Implement the BaseEvaluator interface
3. Add evaluator metadata

Example:
```python
class NewEvaluator(BaseEvaluator):
    @classmethod
    def get_metadata(cls) -> EvaluatorMetadata:
        return EvaluatorMetadata(
            name="New Test",
            description="Test description",
            version="1.0.0",
            category="Category",
            tags=["tag1", "tag2"]
        )
```

## Troubleshooting

If you encounter import errors after installation:
1. Ensure you're running Python 3.8 or higher:
```bash
python --version
```

2. Verify the installation:
```bash
pip list | grep framework
```

3. If issues persist, try reinstalling:
```bash
pip uninstall framework
pip install -e .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

