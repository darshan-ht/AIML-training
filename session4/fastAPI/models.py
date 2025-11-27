"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    
    prompt: str = Field(..., description="Input text prompt for generation")
    max_new_tokens: Optional[int] = Field(None, ge=1, le=2048, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    do_sample: bool = Field(True, description="Whether to use sampling or greedy decoding")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What is the capital of France?",
                "max_new_tokens": 100,
                "temperature": 0.7
            }
        }


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    
    generated_text: str = Field(..., description="Generated text output")
    prompt: str = Field(..., description="Original input prompt")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    
    class Config:
        json_schema_extra = {
            "example": {
                "generated_text": "The capital of France is Paris.",
                "prompt": "What is the capital of France?",
                "tokens_generated": 8
            }
        }


class ModelInfo(BaseModel):
    """Model information and configuration."""
    
    model_name: str = Field(..., description="Name of the base model")
    lora_adapter_path: str = Field(..., description="Path to LoRA adapters")
    device: str = Field(..., description="Device being used for inference")
    model_loaded: bool = Field(..., description="Whether the model is successfully loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "meta-llama/Llama-3.2-1B",
                "lora_adapter_path": "./lora_adapter",
                "device": "cuda",
                "model_loaded": True
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True
            }
        }
