"""
Simple test client for the LLaMA LoRA Inference API.
"""
import requests
import json
import sys


class LLMClient:
    """Client for interacting with the LLM API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self):
        """Get model information."""
        response = requests.get(f"{self.base_url}/model-info")
        response.raise_for_status()
        return response.json()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True
    ):
        """Generate text from a prompt."""
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json()


def main():
    """Run example usage."""
    client = LLMClient()
    
    print("🔍 Checking API health...")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Model loaded: {health['model_loaded']}")
    print()
    
    print("📊 Getting model information...")
    model_info = client.get_model_info()
    print(json.dumps(model_info, indent=2))
    print()
    
    print("🤖 Generating text...")
    prompt = "What is machine learning?"
    print(f"Prompt: {prompt}")
    print()
    
    result = client.generate(
        prompt=prompt,
        max_new_tokens=150,
        temperature=0.7
    )
    
    print(f"Generated text ({result['tokens_generated']} tokens):")
    print(result['generated_text'])
    print()


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API. Is it running?")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
