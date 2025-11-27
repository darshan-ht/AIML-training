"""
Configuration management for the FastAPI LLM application.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model Configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    lora_adapter_path: str = "./lora_adapter"
    huggingface_token: Optional[str] = "YOUR_HF_TOKEN"
    
    # Device Configuration
    device: str = "auto"  # "auto", "cuda", "mps", or "cpu"
    
    # Generation Defaults
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list = ["*"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
