#!/usr/bin/env python3
"""
Run Command: HF_TOKEN=YOUR_TOKEN MODEL_NAME=meta-llama/Llama-3.2-1B python3 sessions/session2-pipelines.py
"""
import os
from transformers import pipeline
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B")
print(f"\nLoading model: {MODEL_NAME}")
generator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    token=HF_TOKEN,
    device_map="auto" 
)
prompt = "What is Machine Learning?"
print(f"Prompt: {prompt}\n")
output = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
print("Output:")
print(output)

