"""
FastAPI application for serving LLaMA model with LoRA adapters.
"""
import logging
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from contextlib import asynccontextmanager

from config import settings
from models import GenerateRequest, GenerateResponse, ModelInfo, HealthResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_loaded = False


def get_device():
    """Detect and return the best available device."""
    if settings.device != "auto":
        logger.info(f"Using manually specified device: {settings.device}")
        return settings.device
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("MPS (Apple Silicon) available")
    else:
        device = "cpu"
        logger.info("Using CPU (no GPU acceleration available)")
    
    return device


def load_model():
    """Load the base model and LoRA adapters."""
    global model, tokenizer, model_loaded
    
    try:
        # Determine device
        device = get_device()
        logger.info(f"Target device: {device}")
        
        # Determine dtype based on device
        if device == "mps":
            # MPS works best with float16
            dtype = torch.float16
            device_map = None  # MPS doesn't support device_map
        elif device == "cuda":
            dtype = torch.float16
            device_map = "auto"
        else:  # CPU
            dtype = torch.float32  # CPU works better with float32
            device_map = None
        
        logger.info(f"Loading tokenizer from {settings.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            settings.model_name,
            token=settings.huggingface_token,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading base model from {settings.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            token=settings.huggingface_token,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        # Move to device if not using device_map
        if device_map is None:
            base_model = base_model.to(device)
        
        logger.info(f"Loading LoRA adapters from {settings.lora_adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            settings.lora_adapter_path,
        )
        
        # Merge LoRA weights for faster inference
        logger.info("Merging LoRA weights with base model")
        model = model.merge_and_unload()
        
        # Ensure model is on correct device
        if device_map is None:
            model = model.to(device)
        
        model.eval()
        model_loaded = True
        
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded = False
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting FastAPI application")
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application")


# Initialize FastAPI app
app = FastAPI(
    title="LLaMA LoRA Inference API",
    description="FastAPI application for serving LLaMA-3.2-1B model with LoRA adapters",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "LLaMA LoRA Inference API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded
    )


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model information and configuration."""
    device_str = str(next(model.parameters()).device) if model_loaded else "unknown"
    
    return ModelInfo(
        model_name=settings.model_name,
        lora_adapter_path=settings.lora_adapter_path,
        device=device_str,
        model_loaded=model_loaded
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_text(request: GenerateRequest):
    """Generate text based on the input prompt."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare generation parameters
        gen_params = {
            "max_new_tokens": request.max_new_tokens or settings.max_new_tokens,
            "temperature": request.temperature or settings.temperature,
            "do_sample": request.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")
        
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_params
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text
        if generated_text.startswith(request.prompt):
            generated_text = generated_text[len(request.prompt):].strip()
        
        tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
        
        logger.info(f"Generated {tokens_generated} tokens")
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            tokens_generated=tokens_generated
        )
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False
    )
