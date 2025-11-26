import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

# Get configuration from environment
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./lora_adapter")
DATASET_PATH = os.getenv("DATASET_PATH", "./dataset")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# Use MPS for Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(device)

# Improved LoRA config
# lora_config = LoraConfig(
#     r=32,  # Increased rank for better capacity
#     lora_alpha=64,  # Increased alpha
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # More modules
#     lora_dropout=0.05,  # Reduced dropout
#     bias="none",
#     task_type="CAUSAL_LM"
# )
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_from_disk(DATASET_PATH)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=94)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# Training Arguments
training_args = TrainingArguments(
    output_dir=LORA_ADAPTER_PATH,
    num_train_epochs=15,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    weight_decay=0.01,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    group_by_length=True,
    save_steps=500,
    logging_steps=50,
    save_total_limit=2,
    remove_unused_columns=True,
    prediction_loss_only=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
trainer.train()
model.save_pretrained(LORA_ADAPTER_PATH)
print(f"Improved training completed. Adapter saved to {LORA_ADAPTER_PATH}")
