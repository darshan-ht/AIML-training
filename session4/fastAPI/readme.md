# LLaMA LoRA Inference API

A production-ready FastAPI application for serving the meta-llama/Llama-3.2-1B model with LoRA adapters. This application provides a RESTful API for text generation with support for Docker deployment and GPU acceleration.

## Features

- 🚀 FastAPI-based REST API
- 🤖 LLaMA-3.2-1B model with LoRA adapters
- 🐳 Docker and Docker Compose support
- 🎮 Multi-platform GPU support (NVIDIA CUDA, Apple MPS)
- 💻 Automatic device detection (CUDA/MPS/CPU)
- 🔧 Configurable generation parameters
- 📊 Health checks and model information endpoints
- 🔒 Environment-based configuration

## Prerequisites

- Python 3.10+
- GPU (optional): NVIDIA GPU with CUDA support OR Apple Silicon Mac with MPS
- Docker and Docker Compose (for containerized deployment)
- HuggingFace account with access to meta-llama/Llama-3.2-1B
- LoRA adapters trained for the model

## Project Structure

```
APP/
├── main.py                 # FastAPI application
├── models.py              # Pydantic models for request/response
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── Dockerfile            # Universal Docker image (CPU/MPS)
├── Dockerfile.cuda       # CUDA-optimized Docker image
├── docker-compose.yml    # Docker Compose configuration
├── .env.example          # Environment variable template
├── .dockerignore         # Docker build exclusions
└── lora_adapter/         # Your LoRA adapter files (not included)
```

## Installation

### Local Development

1. **Clone or navigate to the project directory**

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your HuggingFace token
   ```

5. **Ensure LoRA adapters are in place**
   - Place your LoRA adapter files in the `./lora_adapter` directory
   - The directory should contain `adapter_config.json` and `adapter_model.safetensors` (or `.bin`)

### Docker Deployment

1. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your HuggingFace token
   ```

2. **Ensure LoRA adapters are in place**
   - Place your LoRA adapter files in the `./lora_adapter` directory

3. **Build and run with Docker Compose**
   
   **For CPU or MPS (Apple Silicon):**
   ```bash
   docker-compose up -d
   ```
   
   **For NVIDIA CUDA GPUs:**
   ```bash
   docker-compose --profile cuda up -d llm-api-cuda
   ```
   
   **Or build manually:**
   
   *Universal (CPU/MPS):*
   ```bash
   docker build -t llm-api .
   docker run -d \
     --name llm-api \
     -p 8000:8000 \
     -v $(pwd)/lora_adapter:/app/lora_adapter:ro \
     -e HUGGINGFACE_TOKEN=your_token_here \
     llm-api
   ```
   
   *CUDA-optimized:*
   ```bash
   docker build -f Dockerfile.cuda -t llm-api-cuda .
   docker run -d \
     --name llm-api \
     --gpus all \
     -p 8000:8000 \
     -v $(pwd)/lora_adapter:/app/lora_adapter:ro \
     -e HUGGINGFACE_TOKEN=your_token_here \
     llm-api-cuda
   ```

## Configuration

Edit the `.env` file to configure the application:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | HuggingFace model identifier | `meta-llama/Llama-3.2-1B` |
| `LORA_ADAPTER_PATH` | Path to LoRA adapters | `./lora_adapter` |
| `HUGGINGFACE_TOKEN` | HuggingFace API token | Required |
| `DEVICE` | Device for inference (auto/cuda/mps/cpu) | `auto` |
| `MAX_NEW_TOKENS` | Default max tokens to generate | `512` |
| `TEMPERATURE` | Default sampling temperature | `0.7` |
| `TOP_P` | Default nucleus sampling parameter | `0.9` |
| `TOP_K` | Default top-k sampling parameter | `50` |
| `REPETITION_PENALTY` | Default repetition penalty | `1.1` |
| `API_HOST` | API host address | `0.0.0.0` |
| `API_PORT` | API port | `8000` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Usage

### Starting the Application

**Local:**
```bash
python main.py
# Or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Docker:**
```bash
docker-compose up -d
```

### API Endpoints

#### 1. Root Endpoint
```bash
curl http://localhost:8000/
```

#### 2. Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 3. Model Information
```bash
curl http://localhost:8000/model-info
```

Response:
```json
{
  "model_name": "meta-llama/Llama-3.2-1B",
  "lora_adapter_path": "./lora_adapter",
  "device": "cuda:0",
  "model_loaded": true
}
```

#### 4. Text Generation

**Using curl:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

**Using Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "What is the capital of France?",
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
)

result = response.json()
print(result["generated_text"])
```

**Request Parameters:**
- `prompt` (required): Input text prompt
- `max_new_tokens` (optional): Maximum tokens to generate (1-2048)
- `temperature` (optional): Sampling temperature (0.0-2.0)
- `top_p` (optional): Nucleus sampling probability (0.0-1.0)
- `top_k` (optional): Top-k sampling parameter
- `repetition_penalty` (optional): Repetition penalty (1.0-2.0)
- `do_sample` (optional): Use sampling vs greedy decoding (default: true)

**Response:**
```json
{
  "generated_text": "The capital of France is Paris.",
  "prompt": "What is the capital of France?",
  "tokens_generated": 8
}
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation with interactive API testing.

## Monitoring

### View Logs

**Local:**
```bash
# Application logs are printed to stdout
```

**Docker:**
```bash
docker-compose logs -f llm-api
```

### Check Container Status
```bash
docker-compose ps
```

### Check GPU Usage
```bash
nvidia-smi
```

## Troubleshooting

### Model Not Loading

1. **Check HuggingFace token:**
   - Ensure your token has access to meta-llama/Llama-3.2-1B
   - Verify the token is correctly set in `.env`

2. **Check LoRA adapter path:**
   - Ensure `./lora_adapter` contains the adapter files
   - Verify file permissions

### GPU not detected:
- Run `nvidia-smi` to verify GPU availability (NVIDIA)
- Run `python -c "import torch; print(torch.backends.mps.is_available())"` to check MPS (Apple Silicon)
- Check NVIDIA Docker runtime installation (for CUDA)
- Set `DEVICE=cpu` in `.env` for CPU-only inference
- Set `DEVICE=mps` to force MPS on Apple Silicon

### Out of Memory Errors

1. **Reduce batch size or max tokens:**
   - Lower `max_new_tokens` in requests
   - Use smaller generation parameters

2. **Use CPU inference:**
   - Set `DEVICE=cpu` in `.env`

### Docker Issues

1. **GPU not accessible in container:**
   ```bash
   # Install NVIDIA Container Toolkit
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   ```

2. **Permission denied errors:**
   ```bash
   # Ensure LoRA adapter files are readable
   chmod -R 755 ./lora_adapter
   ```

## Development

### Running Tests
```bash
# Add your test commands here
pytest tests/
```

### Code Formatting
```bash
black .
isort .
```

## Performance Optimization

1. **Model Quantization:** Consider using 8-bit or 4-bit quantization for faster inference
2. **Batch Processing:** Implement batch generation for multiple prompts
3. **Caching:** Cache model outputs for common prompts
4. **Load Balancing:** Use multiple containers behind a load balancer for high traffic

## Security Considerations

- Never commit `.env` file with actual tokens
- Use secrets management in production (e.g., AWS Secrets Manager, HashiCorp Vault)
- Implement rate limiting for production deployments
- Add authentication/authorization for API endpoints
- Use HTTPS in production

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs for error messages
- Open an issue in the repository
