# api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
from datetime import datetime

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from framework.core.registry import Registry
from framework.utils.prompts import load_prompts, PromptCollection

app = FastAPI()

# CORS configuration allowing React + Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=False,  # Change this to False
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize framework
Registry.discover_evaluators()
Registry.discover_llm_interfaces()


class TestRequest(BaseModel):
    model_type: str
    configuration: Dict[str, Any]
    prompt: str
    selected_tests: List[str]

    model_config = {
        'protected_namespaces': ()
    }


class PromptLoadRequest(BaseModel):
    file_path: str


@app.get("/api/evaluators")
async def get_evaluators():
    """Get available evaluators with metadata"""
    evaluators = Registry.list_evaluators()
    evaluator_info = []

    for name in evaluators:
        evaluator = Registry.get_evaluator(name)
        metadata = evaluator.get_metadata()
        evaluator_info.append({
            "id": name,
            "name": metadata.name,
            "description": metadata.description,
            "version": metadata.version,
            "category": metadata.category,
            "tags": metadata.tags
        })

    return {
        "success": True,
        "data": evaluator_info
    }


@app.get("/api/models")
async def get_models():
    """Get available models with configuration options"""
    models = Registry.list_llms()
    model_info = []

    for model_name in models:
        model_class = Registry._llm_interfaces[model_name]
        model_info.append({
            "id": model_name,
            "name": model_class.get_name(),
            "configuration_options": model_class.get_configuration_schema()
        })

    return {
        "success": True,
        "data": model_info
    }


@app.post("/api/prompts/load")
async def load_prompts_file(request: PromptLoadRequest):
    """Load prompts from file"""
    try:
        prompts = load_prompts(request.file_path)
        categories = prompts.get_categories()
        prompt_data = {}

        for category in categories:
            category_prompts = prompts.get_prompts([category])
            prompt_data[category] = [
                {
                    "id": i,
                    "category": category,
                    "text": p.text
                }
                for i, p in enumerate(category_prompts)
            ]

        return {
            "success": True,
            "data": {
                "categories": categories,
                "prompts": prompt_data
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"success": False, "error": str(e)}
        )


@app.post("/api/test")
async def run_test(request: TestRequest):
    """Run tests on a prompt"""
    try:
        llm = Registry.get_llm(request.model_type)

        for key, value in request.configuration.items():
            setattr(llm, key, value)

        response = llm.generate_response(request.prompt)
        results = {}

        for test_name in request.selected_tests:
            evaluator = Registry.get_evaluator(test_name)
            evaluation = evaluator.evaluate(response)
            interpretation = evaluator.interpret(evaluation)

            results[test_name] = {
                "raw_results": evaluation,
                "interpretation": interpretation,
                "summary": {
                    "score": evaluation.get("agency_score", 0) if "agency_score" in evaluation else None,
                    "risk_level": "High" if evaluation.get("agency_score", 0) > 50 else
                    "Medium" if evaluation.get("agency_score", 0) > 20 else "Low",
                    "key_findings": interpretation.split(". ")[:3]
                }
            }

        return {
            "success": True,
            "data": {
                "prompt": request.prompt,
                "response": response,
                "results": results,
                "metadata": {
                    "model_type": request.model_type,
                    "configuration": request.configuration,
                    "timestamp": datetime.now().isoformat()
                }
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)