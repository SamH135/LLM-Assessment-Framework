# LLM Testing Framework

A framework for evaluating and testing Large Language Models (LLMs) across various dimensions including agency detection (the framework is modular and can easily accept new evaluators). Features both a CLI interface and a FastAPI server for API access to all testing functionality.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bwinson/SE-Project.git
cd SE-Project
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

3. Install framework dependencies:
```bash
# Install core dependencies
pip install -r requirements-core.txt

# Install API server dependencies
pip install -r requirements-api.txt
```

if the requirements files don't work, clear the dependencies and try the setup.py
```bash
pip install -e .
```

Core dependencies include:
- transformers (≥4.30.0)
- torch (≥2.0.0)
- numpy (≥1.24.0)
- tokenizers (≥0.13.0)
- tqdm (≥4.65.0)
- datasets (≥2.10.0)
- huggingface-hub (≥0.16.0)
- requests (≥2.32.2)

API Server dependencies include:
- fastapi (==0.68.0)
- uvicorn (==0.15.0)
- pydantic (==1.8.2)
- python-multipart (==0.0.5)
- starlette (==0.14.2)

## Project Structure
```
framework/
├── core/               # Core framework components
│   ├── base.py        # Base classes and interfaces
│   └── registry.py    # Component registry
├── evaluators/         # Test evaluators
│   └── agency/        # Agency detection
├── interfaces/         # LLM interfaces
│   └── llm/          
│       ├── api.py     # Generic API interface
│       └── huggingface.py
├── utils/             # Utility functions
│   └── prompts.py    # Prompt management
└── api_server.py      # FastAPI server
```

## Usage

### CLI Interface
Run the framework from the command line:
```bash
python run.py
```

### API Server
Start the FastAPI server:
```bash
uvicorn api_server:app --reload --port 8000
```
or (the one I use):
```bash
python api_server.py
```


The API server provides the following endpoints:

#### GET /api/evaluators
Get available evaluators with metadata.

Response:
```json
{
    "success": true,
    "data": [{
        "id": "Agency Analysis",
        "name": "Agency Analysis",
        "description": "Evaluates the level of agency expressed in AI responses",
        "version": "1.0.0",
        "category": "Safety",
        "tags": ["agency", "safety", "boundaries", "capabilities"]
    }]
}
```

#### GET /api/models
Get available models with configuration options.

Response:
```json
{
    "success": true,
    "data": [{
        "id": "HuggingFace Model",
        "name": "HuggingFace Model",
        "configuration_options": {
            "model_name": {
                "type": "string",
                "description": "HuggingFace model identifier",
                "default": "gpt2",
                "examples": ["gpt2", "facebook/opt-350m", "EleutherAI/gpt-neo-125M"]
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum length of generated response",
                "default": 100,
                "minimum": 10,
                "maximum": 1000
            }
        }
    }]
}
```

#### POST /api/prompts/load
Load prompts from a file.

Request:
```json
{
    "file_path": "path/to/prompts.txt"
}
```

Response:
```json
{
    "success": true,
    "data": {
        "categories": ["Category1", "Category2"],
        "prompts": {
            "Category1": [
                {
                    "id": 0,
                    "category": "Category1",
                    "text": "prompt1"
                }
            ]
        }
    }
}
```

#### POST /api/test
Run selected tests on a prompt.

Request:
```json
{
    "model_type": "HuggingFace Model",
    "configuration": {
        "model_name": "gpt2",
        "max_length": 50
    },
    "prompt": "What is the meaning of life?",
    "selected_tests": ["Agency Analysis"]
}
```

Response:
```json
{
    "success": true,
    "data": {
        "prompt": "What is the meaning of life?",
        "response": "Life is a journey of discovery...",
        "results": {
            "Agency Analysis": {
                "raw_results": {
                    "agency_score": 0,
                    "high_agency_phrases": []
                },
                "interpretation": "GOOD: Low agency detected...",
                "summary": {
                    "score": 0,
                    "risk_level": "Low",
                    "key_findings": [
                        "GOOD: Low agency detected",
                        "Agency score: 0.00"
                    ]
                }
            }
        }
    }
}
```

For detailed API documentation, see [API_documentation.md](API_documentation.md).

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

### Framework Issues
1. Ensure Python 3.8+ is installed:
```bash
python --version
```

2. Verify all dependencies are installed:
```bash
pip list
```

### API Server Issues
1. Check API server logs:
```bash
uvicorn api_server:app --log-level debug
```

2. Verify API server is running:
```bash
curl http://localhost:8000/api/evaluators
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

