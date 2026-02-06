"""Configuration management for the multi-agent system."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration loaded from environment variables."""
    
    # LLM Settings
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o"))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0")))
    
    # Tavily Search
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    
    # Database
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./data.db"))
    
    # File System
    allowed_file_path: str = field(default_factory=lambda: os.getenv("ALLOWED_FILE_PATH", os.getcwd()))
    
    # Code Execution
    code_execution_timeout: int = field(default_factory=lambda: int(os.getenv("CODE_EXECUTION_TIMEOUT", "30")))
    
    def validate(self) -> list[str]:
        """Validate required configuration. Returns list of missing keys."""
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        return missing


# Global config instance
config = Config()
