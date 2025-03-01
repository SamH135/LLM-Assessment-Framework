# framework/interfaces/llm/bedrock.py

from typing import Optional, Dict, Any, List
import boto3
import json
import time
from datetime import datetime, timedelta
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from ...core.base import BaseLLMInterface


class BedrockLLM(BaseLLMInterface):
    """
    Interface for AWS Bedrock models with enhanced error handling, rate limiting,
    and model-specific configurations.

    Example:
        bedrock = BedrockLLM(
            model_id="anthropic.claude-v2",
            aws_region="us-west-2",
            max_tokens=1000,
            temperature=0.7
        )

        response = bedrock.generate_response(
            prompt="Explain quantum computing",
            stream=False,
            timeout=60
        )
    """

    # Model-specific parameter constraints
    MODEL_PARAMS = {
        "anthropic.claude": {
            "max_tokens": (1, 100000),
            "temperature": (0.0, 1.0),
            "top_p": (0.0, 1.0),
            "top_k": (0, 500),
        },
        "meta.llama": {
            "max_tokens": (1, 4096),
            "temperature": (0.0, 1.0),
            "top_p": (0.0, 1.0),
        },
        "ai21.j2": {
            "max_tokens": (1, 8191),
            "temperature": (0.0, 1.0),
            "top_p": (0.0, 1.0),
        },
        "amazon.titan": {
            "max_tokens": (1, 4096),
            "temperature": (0.0, 1.0),
            "top_p": (0.0, 1.0),
        }
    }

    def __init__(
            self,
            model_id: str,
            aws_region: str,
            aws_access_key_id: Optional[str] = None,
            aws_secret_access_key: Optional[str] = None,
            aws_session_token: Optional[str] = None,
            max_tokens: int = 512,
            temperature: float = 0.7,
            requests_per_second: int = 10,
            max_retries: int = 3,
            timeout: int = 30
    ):
        """
        Initialize Bedrock interface with enhanced configuration.

        Args:
            model_id: Bedrock model identifier
            aws_region: AWS region where Bedrock is deployed
            aws_access_key_id: Optional AWS IAM access key ID
            aws_secret_access_key: Optional AWS IAM secret access key
            aws_session_token: Optional AWS session token
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            requests_per_second: Rate limit for API requests
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = min(max(temperature, 0), 1)

        # Rate limiting settings
        self._requests_per_second = requests_per_second
        self._min_request_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0

        # Initialize AWS client with retry config
        try:
            client_config = Config(
                retries=dict(
                    max_attempts=max_retries,
                    mode='standard'
                ),
                connect_timeout=timeout,
                read_timeout=timeout
            )

            kwargs = {
                'service_name': 'bedrock-runtime',
                'region_name': aws_region,
                'config': client_config
            }

            if aws_access_key_id and aws_secret_access_key:
                kwargs.update({
                    'aws_access_key_id': aws_access_key_id,
                    'aws_secret_access_key': aws_secret_access_key
                })
                if aws_session_token:
                    kwargs['aws_session_token'] = aws_session_token

            self.bedrock = boto3.client(**kwargs)
            self._validate_model_access()

        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"Failed to initialize Bedrock client: {str(e)}")

    def _validate_model_access(self) -> None:
        """Validate model access and permissions."""
        try:
            # Check if model exists
            models = self.bedrock.list_foundation_models()
            available_models = [m['modelId'] for m in models['modelSummaries']]

            if self.model_id not in available_models:
                raise ValueError(
                    f"Model {self.model_id} not available. Verify model access "
                    f"in AWS Bedrock console and ensure proper permissions."
                )

            # Verify invoke permissions with minimal test request
            test_body = self._build_request_body("test", max_tokens=1)
            self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(test_body),
                accept='application/json',
                contentType='application/json'
            )

        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDeniedException':
                raise ValueError(f"No invoke permissions for model {self.model_id}")
            raise RuntimeError(f"Failed to validate model access: {str(e)}")

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration and request template."""
        # Anthropic Claude models
        if "anthropic.claude" in self.model_id.lower():
            return {
                "request_template": {
                    "prompt": "\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": "max_tokens",
                    "temperature": "temperature",
                    "stop_sequences": ["\n\nHuman:"]
                },
                "response_key": "completion",
                "streaming_supported": True
            }

        # Meta Llama models
        elif "meta.llama" in self.model_id.lower():
            return {
                "request_template": {
                    "prompt": "{prompt}",
                    "max_gen_len": "max_tokens",
                    "temperature": "temperature"
                },
                "response_key": "generation",
                "streaming_supported": True
            }

        # AI21 Jurassic models
        elif "ai21.j2" in self.model_id.lower():
            return {
                "request_template": {
                    "prompt": "{prompt}",
                    "maxTokens": "max_tokens",
                    "temperature": "temperature",
                    "stopSequences": ["\n"]
                },
                "response_key": "completions[0].data.text",
                "streaming_supported": False
            }

        # Amazon Titan models
        elif "amazon.titan" in self.model_id.lower():
            return {
                "request_template": {
                    "inputText": "{prompt}",
                    "textGenerationConfig": {
                        "maxTokenCount": "max_tokens",
                        "temperature": "temperature",
                        "stopSequences": ["\n"]
                    }
                },
                "response_key": "results[0].outputText",
                "streaming_supported": True
            }

        else:
            raise ValueError(
                f"Unsupported model: {self.model_id}. "
                "Please verify model compatibility."
            )

    @classmethod
    def get_configuration_schema(cls) -> Dict[str, Any]:
        return {
            "model_id": {
                "type": "string",
                "description": "Bedrock model identifier",
                "required": True,
                "examples": ["anthropic.claude-v2", "meta.llama2-70b"]
            },
            "aws_region": {
                "type": "string",
                "description": "AWS region where Bedrock is deployed",
                "required": True,
                "examples": ["us-west-2", "us-east-1"]
            },
            "max_tokens": {
                "type": "number",
                "description": "Maximum tokens in response",
                "default": 512,
                "min": 1,
                "max": 4096
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature (0-1)",
                "default": 0.7,
                "min": 0,
                "max": 1
            },
            "aws_access_key_id": {
                "type": "string",
                "description": "AWS access key ID (optional if using IAM roles)",
                "required": False,
                "sensitive": True
            },
            "aws_secret_access_key": {
                "type": "string",
                "description": "AWS secret access key",
                "required": False,
                "sensitive": True
            }
        }

    def _validate_parameters(self, **kwargs) -> None:
        """Validate model-specific parameters."""
        for model_type, constraints in self.MODEL_PARAMS.items():
            if model_type in self.model_id.lower():
                for param, (min_val, max_val) in constraints.items():
                    if param in kwargs:
                        value = kwargs[param]
                        if not min_val <= value <= max_val:
                            raise ValueError(
                                f"Parameter {param} must be between {min_val} and {max_val}"
                            )

    def _build_request_body(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Build request body using model template and parameters."""
        model_config = self._get_model_config()
        request_body = {}
        template = model_config["request_template"]

        for key, value in template.items():
            if isinstance(value, str) and value in kwargs:
                request_body[key] = kwargs[value]
            elif value == "max_tokens":
                request_body[key] = kwargs.get('max_tokens', self.max_tokens)
            elif value == "temperature":
                request_body[key] = kwargs.get('temperature', self.temperature)
            else:
                request_body[key] = value.format(prompt=prompt) if isinstance(value, str) else value

        return request_body

    def _extract_response(self, response_body: Dict[str, Any], response_key: str) -> str:
        """Extract response text from nested dictionary using dot notation key."""
        try:
            keys = response_key.split('.')
            value = response_body
            for key in keys:
                if '[' in key:
                    key, index = key.split('[')
                    index = int(index.rstrip(']'))
                    value = value[key][index]
                else:
                    value = value[key]
            return str(value)
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Failed to extract response using key '{response_key}': {str(e)}")

    def _handle_rate_limit(self) -> None:
        """Implement rate limiting for API requests."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time

        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def generate_response(self, prompt: str, stream: bool = False, **kwargs) -> str:
        """
        Generate response using Bedrock model with enhanced error handling and rate limiting.

        Args:
            prompt: Input text prompt
            stream: Whether to use streaming response (if supported)
            **kwargs: Optional parameters (max_tokens, temperature, etc.)

        Returns:
            Generated text response

        Raises:
            ValueError: For invalid parameters
            RuntimeError: For API errors
        """
        try:
            # Validate parameters
            self._validate_parameters(**kwargs)

            # Apply rate limiting
            self._handle_rate_limit()

            # Get model configuration and build request
            model_config = self._get_model_config()
            request_body = self._build_request_body(prompt, **kwargs)

            # Handle streaming if supported
            if stream and model_config.get("streaming_supported", False):
                response = self.bedrock.invoke_model_with_response_stream(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    accept='application/json',
                    contentType='application/json'
                )

                full_response = ""
                for event in response['body']:
                    chunk = json.loads(event['chunk']['bytes'].decode('utf-8'))
                    full_response += self._extract_response(chunk, model_config["response_key"])
                return full_response

            else:
                # Standard synchronous request
                response = self.bedrock.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request_body),
                    accept='application/json',
                    contentType='application/json'
                )

                if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                    raise RuntimeError(
                        f"API call failed with status {response['ResponseMetadata']['HTTPStatusCode']}"
                    )

                response_body = json.loads(response['body'].read().decode('utf-8'))
                return self._extract_response(response_body, model_config["response_key"])

        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"AWS API error: {str(e)}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse response: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {str(e)}")

    @classmethod
    def get_name(cls) -> str:
        return "AWS Bedrock Model"