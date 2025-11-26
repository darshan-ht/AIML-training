from datasets import Dataset
import json
import os

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        # Create sample JSONL if file doesn't exist
        sample_data = [
            {"instruction": "What is Python?", "output": "Python is a programming language."},
            {"instruction": "How to print hello?", "output": "Use print('hello') in Python."},
            {"instruction": "What is AI?", "output": "AI is artificial intelligence technology."}
        ]
        with open(file_path, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        print(f"Created sample {file_path}")
        data = sample_data
    return data

def format_prompt(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

# Get configuration from environment
DATA_FILE = os.getenv("DATA_FILE", "data.jsonl")
DATASET_PATH = os.getenv("DATASET_PATH", "./dataset")

# Load data from JSONL
data = load_jsonl(DATA_FILE)

# Create dataset
dataset = Dataset.from_list(data)
dataset = dataset.map(lambda x: {"text": format_prompt(x)})

# Save
dataset.save_to_disk(DATASET_PATH)
print(f"Dataset prepared with {len(data)} examples and saved to {DATASET_PATH}")
