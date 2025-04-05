# framework/interfaces/llm/ai_agent.py
from typing import Dict, Any, Optional, List, Tuple
import importlib.util
import os
import sys
import tempfile
import logging
import json
import re
import requests
from pathlib import Path
from ...core.base import BaseLLMInterface
from ...core.registry import Registry

# Try to import API client libraries that might be available
LLM_CLIENTS_AVAILABLE = {}
try:
    import anthropic

    LLM_CLIENTS_AVAILABLE["anthropic"] = True
except ImportError:
    LLM_CLIENTS_AVAILABLE["anthropic"] = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocsSearchEngine:
    """Search engine for finding and parsing API documentation."""

    def __init__(self, serper_api_key: Optional[str] = None):
        """
        Initialize the search engine.

        Args:
            serper_api_key: API key for Serper.dev (Google Search API)
        """
        self.serper_api_key = serper_api_key or os.environ.get("SERPER_API_KEY")
        self.search_cache = {}

    def search_api_docs(self, provider_name: str) -> Dict[str, Any]:
        """
        Search for API documentation for the provider.

        Args:
            provider_name: Name of the LLM provider

        Returns:
            Dictionary with search results including URLs and snippets
        """
        if not provider_name:
            return {"provider": "", "results": []}

        cache_key = f"api_docs:{provider_name.lower()}"
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        # Define search queries
        search_queries = [
            f"{provider_name} API documentation",
            f"{provider_name} LLM API reference",
            f"{provider_name} text generation API"
        ]

        all_results = []

        # Try to use Serper.dev (Google Search API) if available
        if self.serper_api_key:
            for query in search_queries:
                try:
                    results = self._search_with_serper(query)
                    if results:
                        all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Error searching with Serper: {e}")
                    continue

        # Fallback to pre-defined documentation URLs if search fails
        if not all_results:
            default_docs = self._get_default_docs_urls(provider_name)
            if default_docs:
                all_results.append({
                    "title": f"{provider_name} API Documentation",
                    "link": default_docs,
                    "snippet": f"Official API documentation for {provider_name}"
                })

        # Cache the results
        result_dict = {
            "provider": provider_name,
            "results": all_results
        }
        self.search_cache[cache_key] = result_dict

        return result_dict

    def _search_with_serper(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using Serper.dev API.

        Args:
            query: Search query

        Returns:
            List of search result dictionaries
        """
        if not self.serper_api_key:
            return []

        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": query,
            "gl": "us",
            "hl": "en",
            "num": 5
        })
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(url, headers=headers, data=payload, timeout=15)
            response.raise_for_status()

            data = response.json()
            organic_results = data.get("organic", [])

            return [{
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", "")
            } for result in organic_results]
        except requests.RequestException as e:
            logger.warning(f"Request failed in _search_with_serper: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in _search_with_serper: {e}")
            return []

    def _get_default_docs_urls(self, provider_name: str) -> Optional[str]:
        """
        Get default documentation URL for common providers.

        Args:
            provider_name: Name of the LLM provider

        Returns:
            URL string or None
        """
        if not provider_name:
            return None

        provider_lower = provider_name.lower()
        default_urls = {
            "openai": "https://platform.openai.com/docs/api-reference",
            "anthropic": "https://docs.anthropic.com/claude/reference",
            "cohere": "https://docs.cohere.com/reference/about",
            "mistral": "https://docs.mistral.ai/api/",
            "hugging face": "https://huggingface.co/docs/api-inference/index",
            "huggingface": "https://huggingface.co/docs/api-inference/index",
            "ai21": "https://docs.ai21.com/reference/j2-complete-api",
            "ai21 labs": "https://docs.ai21.com/reference/j2-complete-api",
            "google": "https://ai.google.dev/api/rest",
            "google ai": "https://ai.google.dev/api/rest",
            "gemini": "https://ai.google.dev/api/rest",
            "groq": "https://console.groq.com/docs/quickstart",
            "together": "https://docs.together.ai/reference/inference",
            "together ai": "https://docs.together.ai/reference/inference",
            "ollama": "https://github.com/ollama/ollama/blob/main/docs/api.md",
            "replicate": "https://replicate.com/docs/reference/http",
        }

        # Try exact match first
        if provider_lower in default_urls:
            return default_urls[provider_lower]

        # Try partial match
        for key, url in default_urls.items():
            if key in provider_lower or provider_lower in key:
                return url

        return None

    def fetch_api_examples(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch and parse API examples from documentation URL.

        Args:
            url: URL to fetch from

        Returns:
            Dictionary with API examples or None if unsuccessful
        """
        if not url:
            return None

        try:
            # Fetch the URL content
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            # Extract code blocks and example JSON
            content = response.text
            examples = self._extract_api_examples(content, url)

            if examples:
                return examples

            logger.warning(f"Could not extract API examples from {url}")
            return None
        except requests.RequestException as e:
            logger.warning(f"Error fetching API examples: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error in fetch_api_examples: {e}")
            return None

    def _extract_api_examples(self, content: str, url: str) -> Optional[Dict[str, Any]]:
        """
        Extract API examples from HTML content.

        Args:
            content: HTML content
            url: Source URL for reference

        Returns:
            Dictionary with extracted examples or None
        """
        if not content:
            return None

        try:
            # Look for code blocks in HTML
            code_blocks = re.findall(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>', content, re.DOTALL)

            # Also look for markdown code blocks
            markdown_blocks = re.findall(r'```(?:json|python|bash)?\s*(.*?)```', content, re.DOTALL)

            # Combine and clean code blocks
            all_blocks = code_blocks + markdown_blocks
            clean_blocks = [self._clean_code_block(block) for block in all_blocks]

            # Filter for likely API request/response examples
            api_examples = self._filter_api_examples(clean_blocks)

            if api_examples:
                return {
                    "source_url": url,
                    "examples": api_examples
                }

            return None
        except Exception as e:
            logger.warning(f"Error extracting API examples: {e}")
            return None

    def _clean_code_block(self, block: str) -> str:
        """Clean HTML entities and whitespace from code block."""
        if not isinstance(block, str):
            return ""

        try:
            # Replace HTML entities
            block = block.replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&amp;', '&')
            # Normalize whitespace
            block = re.sub(r'\s+', ' ', block).strip()
            return block
        except Exception as e:
            logger.warning(f"Error cleaning code block: {e}")
            return ""

    def _filter_api_examples(self, blocks: List[str]) -> List[Dict[str, Any]]:
        """
        Filter code blocks to find likely API examples.

        Args:
            blocks: List of cleaned code blocks

        Returns:
            List of API example dictionaries
        """
        api_examples = []

        for block in blocks:
            try:
                # Check if block looks like JSON
                if (block.startswith('{') and block.endswith('}')) or (block.startswith('[') and block.endswith(']')):
                    try:
                        json_data = json.loads(block)
                        # Check if it looks like an API request or response
                        if (isinstance(json_data, dict) and
                                any(key in json_data for key in
                                    ['prompt', 'messages', 'model', 'completion', 'choices', 'content'])):
                            api_examples.append({
                                "type": "json",
                                "content": json_data
                            })
                    except json.JSONDecodeError:
                        continue

                # Check if block looks like a curl command
                elif block.startswith('curl ') and ('api' in block.lower()):
                    api_examples.append({
                        "type": "curl",
                        "content": block
                    })

                # Check if block looks like Python API call
                elif any(pattern in block.lower() for pattern in
                         ['.create(', '.generate(', '.completions(', 'api_key']):
                    api_examples.append({
                        "type": "python",
                        "content": block
                    })
            except Exception as e:
                logger.warning(f"Error processing block in _filter_api_examples: {e}")
                continue

        return api_examples


class LLMInterfaceGenerator:
    """
    Uses Claude 3.7 to generate interface code for new model providers.

    This class can:
    1. Analyze a provider's API documentation
    2. Generate appropriate interface code
    3. Test the generated interface
    4. Register the interface with the framework
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 model_name: str = "claude-3-7-sonnet-20250219"):
        """
        Initialize the interface generator with Claude 3.7.

        Args:
            api_key: API key for Anthropic Claude
            api_base: Optional custom API endpoint
            model_name: Which Claude model to use, defaults to 3.7 Sonnet
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.api_base = api_base or "https://api.anthropic.com"
        self.model_name = model_name

        # Check if API key is available
        if not self.api_key:
            raise ValueError(
                "API key for Claude is required. Either pass it as api_key or set ANTHROPIC_API_KEY environment variable.")

        # Initialize Claude client
        self._init_claude_client()

    def _init_claude_client(self):
        """Initialize the Claude client for code generation."""
        self.client = None
        if LLM_CLIENTS_AVAILABLE["anthropic"]:
            logger.info("Using Anthropic Python client for Claude API calls")
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                if self.api_base and self.api_base != "https://api.anthropic.com":
                    self.client.base_url = self.api_base
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}. Will use direct API calls.")
        else:
            logger.info("Anthropic Python client not available, will use direct API calls")

    def _call_claude_api(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call Claude API either through the client library or direct REST API.

        Args:
            system_prompt: System prompt to set context
            user_prompt: User prompt with the specific request

        Returns:
            Generated text response
        """
        try:
            if self.client:
                try:
                    # Use the client library if available
                    message = self.client.messages.create(
                        model=self.model_name,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=4000,
                        temperature=0.2
                    )
                    return message.content[0].text
                except Exception as e:
                    logger.warning(f"Client library error: {str(e)}. Falling back to direct API call.")

            # Direct API call (as fallback or primary method)
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            data = {
                "model": self.model_name,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 4000,
                "temperature": 0.2
            }

            response = requests.post(
                f"{self.api_base}/v1/messages",
                headers=headers,
                json=data,
                timeout=60  # Add timeout
            )

            response.raise_for_status()
            result = response.json()
            if "content" in result and len(result["content"]) > 0 and "text" in result["content"][0]:
                return result["content"][0]["text"]
            else:
                raise ValueError(f"Unexpected response format: {result}")

        except Exception as e:
            logger.error(f"Failed to call Claude API: {str(e)}")
            raise RuntimeError(f"Claude API call failed: {str(e)}")

    # Fix for generate_interface method in LLMInterfaceGenerator class
    def generate_interface(self,
                           provider_name: str,
                           api_docs_url: Optional[str] = None,
                           api_examples: Optional[Dict[str, Any]] = None,
                           output_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate interface code for a given provider using Claude 3.7.

        Args:
            provider_name: Name of the LLM provider
            api_docs_url: URL to the provider's API documentation
            api_examples: Dictionary containing example API calls and responses
            output_path: Path where to save the generated interface

        Returns:
            Tuple of (generated_code, file_path)
        """
        if not provider_name:
            raise ValueError("Provider name must be specified")

        logger.info(f"Generating interface for {provider_name}...")

        # Construct the system prompt
        system_prompt = """You are an expert Python developer specializing in AI model integrations.
    Your task is to create a Python interface file for a testing framework that can connect to various LLM providers.
    The interface must follow a specific structure and should be fully functional with the provider's API.

    The file should implement the BaseLLMInterface abstract class which has the following key methods:
    1. __init__ - Initialize the interface with configuration parameters
    2. generate_response - Generate a response using the provider's API
    3. get_configuration_schema (class method) - Return configuration requirements as a dictionary
    4. get_name (class method) - Return the display name of the interface

    Use defensive programming throughout your implementation, including:
    - Proper error handling with try/except blocks
    - Parameter validation
    - Useful error messages
    - Type hinting for all functions and parameters

    For error handling, make sure to:
    1. Handle API rate limits, connection issues, and authentication errors
    2. Provide useful error messages that help diagnose the problem
    3. Implement retries with exponential backoff for common transient failures
    """

        # Construct the user prompt
        user_prompt = f"""Please create a Python interface file for {provider_name} that follows the BaseLLMInterface structure.

    The interface should include:
    1. Necessary imports for the provider's API
    2. A class named {provider_name.title().replace(' ', '')}Interface that inherits from BaseLLMInterface
    3. Complete implementation of all required methods
    4. Appropriate error handling and validation
    5. Clear documentation

    """

        if api_docs_url:
            user_prompt += f"\nThe API documentation can be found at: {api_docs_url}\n"

        if api_examples:
            user_prompt += "\nHere are some examples of API calls for this provider:\n"
            user_prompt += json.dumps(api_examples, indent=2)

        user_prompt += (
            "The generated code should follow this template structure:\n\n"
            "```python\n"
            "# framework/interfaces/llm/provider_name.py\n"
            "from typing import Dict, Any, Optional\n"
            "# Import provider-specific libraries\n"
            "from ...core.base import BaseLLMInterface\n\n"
            "class ProviderNameInterface(BaseLLMInterface):\n"
            "    \"\"\"Interface for ProviderName API models\"\"\"\n\n"
            "    def __init__(self, api_key: str, ...other params...):\n"
            "        \"\"\"Initialize the interface with configuration\"\"\"\n"
            "        # Initialize parameters\n"
            "        # Set up client if needed\n\n"
            "    @classmethod\n"
            "    def get_configuration_schema(cls) -> Dict[str, Any]:\n"
            "        \"\"\"Return configuration requirements for the interface\"\"\"\n"
            "        return {\n"
            "            # Configuration fields with type, description, etc.\n"
            "        }\n\n"
            "    def generate_response(self, prompt: str, **kwargs) -> str:\n"
            "        \"\"\"Generate response using the provider's API\"\"\"\n"
            "        # Generate and return response\n\n"
            "    @classmethod\n"
            "    def get_name(cls) -> str:\n"
            "        \"\"\"Return the display name of the interface\"\"\"\n"
            "        return \"Provider Name\"\n"
            "```\n"
            "Please provide only the complete, ready-to-use Python code with no additional explanations. "
            "Make sure to implement proper error handling, parameter validation, and appropriate defaults. "
            "Focus on creating a production-ready interface that handles edge cases gracefully.\n"
        )

        # Call Claude to generate the interface code
        try:
            generated_code = self._call_claude_api(system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"Failed to generate interface code: {e}")
            raise RuntimeError(f"Interface generation failed: {e}")

        # Extract code from markdown if necessary
        generated_code = self._extract_code_from_response(generated_code)

        # Save the generated code if output path provided
        file_path = None
        if output_path:
            try:
                file_path = self._save_interface_code(provider_name, generated_code, output_path)
            except Exception as e:
                logger.error(f"Failed to save interface code: {e}")
                raise RuntimeError(f"Failed to save interface code: {e}")

        return generated_code, file_path


    class AIAgentInterface(BaseLLMInterface):
        """
        AI Agent interface that can dynamically generate and handle various LLM providers.
        CopyThis interface can:
        1. Try to connect to models not explicitly supported
        2. Generate interface code for new model providers
        3. Register new interfaces dynamically
        """

    def __init__(self, model_provider: str = None, api_base: str = None, api_key: str = None,
                 model_name: str = None, **kwargs):
        """
        Initialize the AI Agent interface.

        Args:
            model_provider: The LLM provider (e.g., "openai", "anthropic", "cohere")
            api_base: Base URL for the API
            api_key: API key for authentication
            model_name: Specific model to use
            **kwargs: Additional provider-specific parameters
        """
        self.model_provider = model_provider
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.additional_params = kwargs

        # Initialize dynamic interface
        self.interface = None

        # Get Claude API key for interface generation
        self.claude_api_key = os.environ.get("ANTHROPIC_API_KEY")

        # Initialize search engine
        self.search_engine = DocsSearchEngine(serper_api_key=os.environ.get("SERPER_API_KEY"))

        # Initialize dynamic interface
        self._init_dynamic_interface()

    def _init_dynamic_interface(self):
        """
        Initialize the appropriate interface based on the provider.
        """
        if not self.model_provider:
            logger.warning(
                "No model provider specified. The AI Agent will attempt to identify the provider during generation.")
            return

        # Check if we can dynamically load an existing interface
        interface_path = self._get_interface_path()
        if interface_path and os.path.exists(interface_path):
            try:
                self.interface = self._load_existing_interface(interface_path)
                logger.info(f"Successfully loaded existing interface for {self.model_provider}")
                return
            except Exception as e:
                logger.error(f"Failed to load existing interface: {str(e)}")

        # If we couldn't load an existing interface, generate one
        try:
            self._generate_interface()
        except Exception as e:
            logger.error(f"Failed to generate interface: {str(e)}")
            raise RuntimeError(f"Could not initialize interface for {self.model_provider}: {str(e)}")

    def _get_interface_path(self) -> Optional[str]:
        """Get the path to the interface file for the specified provider."""
        if not self.model_provider:
            return None

        # Convert model provider to snake_case filename
        provider_file = self.model_provider.lower().replace(" ", "_").replace("-", "_")
        path = Path(__file__).parent / f"{provider_file}.py"
        return str(path) if path.exists() else None

    def _load_existing_interface(self, interface_path: str) -> BaseLLMInterface:
        """
        Load an existing interface module and instantiate the interface class.

        Args:
            interface_path: Path to the interface Python file

        Returns:
            An instance of the interface class
        """
        if not interface_path or not os.path.exists(interface_path):
            raise ValueError(f"Interface path does not exist: {interface_path}")

        # Extract the module name from the path
        module_name = os.path.basename(interface_path).replace(".py", "")
        full_module_name = f"framework.interfaces.llm.{module_name}"

        # First try to import using standard import
        try:
            module = importlib.import_module(full_module_name)
        except ImportError:
            # If that fails, load the module from the file path
            try:
                spec = importlib.util.spec_from_file_location(module_name, interface_path)
                if spec is None:
                    raise ImportError(f"Failed to create module spec for {interface_path}")

                module = importlib.util.module_from_spec(spec)
                if spec.loader is None:
                    raise ImportError(f"Module spec has no loader for {interface_path}")

                spec.loader.exec_module(module)
            except Exception as e:
                raise ImportError(f"Failed to load module from {interface_path}: {e}")

        # Find the interface class in the module (should be a BaseLLMInterface subclass)
        interface_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                    issubclass(attr, BaseLLMInterface) and
                    attr != BaseLLMInterface):
                interface_class = attr
                break

        if not interface_class:
            raise ValueError(f"No BaseLLMInterface subclass found in {interface_path}")

        # Initialize the interface with our parameters
        interface_params = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "model_name": self.model_name,
        }
        # Add additional parameters
        if self.additional_params:
            interface_params.update(self.additional_params)

        # Filter out parameters that aren't accepted by the interface
        valid_params = {}
        import inspect
        sig = inspect.signature(interface_class.__init__)
        for param in sig.parameters:
            if param in interface_params and param != 'self':
                valid_params[param] = interface_params[param]

        return interface_class(**valid_params)

    def _generate_interface(self):
        """
        Generate a new interface for the specified provider using Claude 3.7.

        This method:
        1. Searches for API documentation
        2. Uses Claude to generate an interface
        3. Saves the interface to a file
        4. Loads and registers the new interface
        """
        if not self.claude_api_key:
            self.claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not self.claude_api_key:
                raise ValueError(
                    "Anthropic API key required for interface generation. Set ANTHROPIC_API_KEY environment variable.")

        # Search for API documentation
        api_docs_url = None
        api_examples = None

        try:
            docs_results = self.search_engine.search_api_docs(self.model_provider)
            if docs_results and 'results' in docs_results and docs_results['results']:
                # Take the top result
                api_docs_url = docs_results['results'][0]['link']
                logger.info(f"Found API documentation at: {api_docs_url}")

                # Try to fetch API examples
                examples_data = self.search_engine.fetch_api_examples(api_docs_url)
                if examples_data and 'examples' in examples_data:
                    api_examples = examples_data['examples']
                    logger.info(f"Found {len(api_examples)} API examples")
        except Exception as e:
            logger.warning(f"Error during API docs search: {e}")

        try:
            # Initialize the interface generator
            generator = LLMInterfaceGenerator(
                api_key=self.claude_api_key
            )

            # Determine the output directory
            output_dir = Path(__file__).parent

            # Generate the interface
            result = generator.generate_and_test(
                provider_name=self.model_provider,
                api_docs_url=api_docs_url,
                api_examples=api_examples,
                test_api_key=self.api_key,
                output_path=output_dir
            )

            if result.get("test_success", False) or not self.api_key:
                # Interface was tested successfully or couldn't be tested
                file_path = result.get("file_path")

                if file_path:
                    try:
                        # Load the newly generated interface
                        self.interface = self._load_existing_interface(file_path)

                        # Register the new interface with the Registry
                        module_name = os.path.basename(file_path).replace(".py", "")
                        full_module_name = f"framework.interfaces.llm.{module_name}"

                        try:
                            # Reload the module if it's already in sys.modules
                            if full_module_name in sys.modules:
                                module = importlib.reload(sys.modules[full_module_name])
                            else:
                                module = importlib.import_module(full_module_name)

                            # Find and register the interface class
                            for attr_name in dir(module):
                                attr = getattr(module, attr_name)
                                if (isinstance(attr, type) and
                                        issubclass(attr, BaseLLMInterface) and
                                        attr != BaseLLMInterface and
                                        attr != AIAgentInterface):
                                    # Register the interface with the Registry
                                    Registry.register_llm(attr)
                                    logger.info(f"Successfully registered new interface: {attr.get_name()}")
                                    break
                        except Exception as e:
                            logger.error(f"Failed to register new interface: {str(e)}")
                    except Exception as e:
                        logger.error(f"Failed to load newly generated interface: {str(e)}")
                        raise RuntimeError(f"Interface generation succeeded but loading failed: {str(e)}")
                else:
                    raise RuntimeError("Interface generation failed: No file path returned")
            else:
                # Interface testing failed
                error_message = result.get("test_result", "Unknown error")
                raise RuntimeError(f"Generated interface failed testing: {error_message}")
        except Exception as e:
            logger.error(f"Interface generation process failed: {str(e)}")
            raise RuntimeError(f"Could not generate interface: {str(e)}")

    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the appropriate LLM interface.

        Args:
            prompt: Input prompt for the LLM
            **kwargs: Additional parameters to pass to the underlying interface

        Returns:
            Generated response text
        """
        if not self.interface:
            if not self.model_provider:
                raise ValueError("No model provider specified and no interface initialized")

            # Try to initialize the interface again
            self._init_dynamic_interface()

            if not self.interface:
                raise RuntimeError(f"Could not initialize interface for {self.model_provider}")

        try:
            return self.interface.generate_response(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"

    @classmethod
    def get_configuration_schema(cls) -> Dict[str, Any]:
        """
        Return configuration requirements for the AI Agent interface.

        Returns:
            Dictionary with configuration fields
        """
        return {
            "model_provider": {
                "type": "string",
                "description": "Name of the LLM provider",
                "required": True,
                "examples": ["OpenAI", "Anthropic", "Cohere", "Mistral"]
            },
            "api_key": {
                "type": "string",
                "description": "API key for authentication",
                "required": True,
                "secret": True
            },
            "api_base": {
                "type": "string",
                "description": "Base URL for the API (optional)",
                "required": False
            },
            "model_name": {
                "type": "string",
                "description": "Specific model to use (optional)",
                "required": False
            }
        }

    @classmethod
    def get_name(cls) -> str:
        """
        Return the display name of the interface.

        Returns:
            Interface display name
        """
        return "AI Agent (Dynamic)"

    def detect_and_create_interface(self, api_key: str, api_base: Optional[str] = None) -> str:
        """
        Attempt to detect the provider from an API key and create an interface.

        Args:
            api_key: API key to analyze
            api_base: Optional API base URL

        Returns:
            Result message
        """
        if not api_key:
            return "Error: API key is required for detection"

        # Simple pattern matching for common API key formats
        provider = None

        if api_key.startswith("sk-"):
            if len(api_key) > 50:
                provider = "OpenAI"
            else:
                provider = "Groq"
        elif api_key.startswith("sk-ant-"):
            provider = "Anthropic"
        elif api_key.startswith("co-"):
            provider = "Cohere"
        elif api_key.startswith("eyJ") and api_base and "mistral" in api_base.lower():
            provider = "Mistral"
        elif api_key.startswith("gsk_"):
            provider = "Gemini"
        elif api_key.startswith("at-"):
            provider = "Together AI"

        if not provider:
            return "Could not detect provider from API key format. Please specify the provider name."

        # Set properties and initialize
        self.model_provider = provider
        self.api_key = api_key
        self.api_base = api_base

        try:
            self._init_dynamic_interface()
            return f"Successfully created interface for {provider}"
        except Exception as e:
            logger.error(f"Failed to create interface: {e}")
            return f"Failed to create interface: {str(e)}"
