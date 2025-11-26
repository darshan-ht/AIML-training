import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Get configuration from environment
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./lora_adapter")

# Load base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(device)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

def generate_response(instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Reduced for focused responses
            temperature=0.1,  # Lower temperature for accuracy
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.2,  # Higher penalty
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,  # Enable proper stopping
            early_stopping=True,
            no_repeat_ngram_size=3  # Prevent repetition
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    if "### Response:\n" in response:
        response = response.split("### Response:\n")[1].strip()
        # Stop at next instruction if present
        if "### Instruction:" in response:
            response = response.split("### Instruction:")[0].strip()
    return response

# Test multiple questions
if __name__ == "__main__":
    test_questions = [
        "What is HeapTrace technology?",
        "What services does HeapTrace provide?",
        "Where is HeapTrace located?",
        "What technologies does HeapTrace use?",
        "Who are HeapTrace’s clients and where are they located?",
        "When was HeapTrace founded?",
        "Who leads HeapTrace Technology?",
        "What are some enterprise solutions offered by HeapTrace?",
        "What is HeapTrace’s approach to DevOps?",
        "What QA and testing services are provided by HeapTrace?",
        "Which industries does HeapTrace have specific case experience in?",
        "What is HeapTrace’s culture and mission?",
        "Which AI/ML services does HeapTrace provide?",
        "Which certifications or awards has HeapTrace achieved?",
    ]
    
    for question in test_questions:
        response = generate_response(question)
        print(f"Q: {question}")
        print(f"A: {response}")
        print("-" * 50)
