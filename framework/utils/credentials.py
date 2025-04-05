# framework/utils/credentials.py
import os
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class CredentialManager:
    """Manages API credentials for various services"""
    
    CONFIG_PATH = Path.home() / ".config" / "llm_framework"
    CREDENTIALS_FILE = CONFIG_PATH / "credentials.json"
    
    @classmethod
    def initialize(cls):
        """Initialize credential storage directory"""
        if not cls.CONFIG_PATH.exists():
            cls.CONFIG_PATH.mkdir(parents=True, exist_ok=True)
            
        if not cls.CREDENTIALS_FILE.exists():
            cls._save_credentials({})
    
    @classmethod
    def get_credential(cls, service_name: str) -> str:
        """Get a credential for the specified service"""
        credentials = cls._load_credentials()
        
        # First check if credential exists in credentials file
        if service_name in credentials:
            return credentials[service_name]
            
        # Then check environment variables
        env_var_name = f"LLM_FRAMEWORK_{service_name.upper()}_KEY"
        if env_var_name in os.environ:
            return os.environ[env_var_name]
            
        return None
        
    @classmethod
    def save_credential(cls, service_name: str, value: str) -> None:
        """Save a credential for the specified service"""
        credentials = cls._load_credentials()
        credentials[service_name] = value
        cls._save_credentials(credentials)
        
    @classmethod
    def _load_credentials(cls) -> dict:
        """Load credentials from file"""
        cls.initialize()
        try:
            with open(cls.CREDENTIALS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return {}
            
    @classmethod
    def _save_credentials(cls, credentials: dict) -> None:
        """Save credentials to file"""
        try:
            with open(cls.CREDENTIALS_FILE, 'w') as f:
                json.dump(credentials, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")