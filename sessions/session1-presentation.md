# Fine Tunning LLM Models
## Hugging Face Basics
**Foundational Concepts**

- Open-source ecosystem for NLP + LLM workflows
- Model hub, datasets, tokenizers, pipelines
- Core libs: transformers, datasets, peft, accelerate
----
## Transformer Architecture
**Core Building Block of Modern LLMs**

- Encoder–decoder
- Positional embeddings maintain token order
- Highly parallelizable training
- Backbone for GPT, LLaMA, T5, Falcon, etc.
----
## PEFT (Parameter-Efficient Finetuning)
**Why PEFT?**

- Finetune large models with minimal compute
- Train only small subset of parameters
- Lower cost, memory, and training time
- Methods: LoRA, Prefix Tuning, Adapters, P-Tuning
----
## LoRA (Low-Rank Adaptation)
**How LoRA Works**

- Injects low-rank trainable matrices
- Freezes original model weights
- Only small matrices updated
- High performance, minimal GPU usage
----
## Tokenizers & Vocabulary
**Foundation for Understanding Text**

- Converts text → tokens → model inputs
- Handles special tokens and vocab extension
- Tokenization errors impact model quality
----
## Pretraining vs Finetuning
1. **Pretraining**
   - Large-scale self-supervised training
   - Builds general language understanding
2. **Finetuning**
   - Task-specific dataset adaptation

----
## LoRA & QLoRA
1. **LoRA**
   - Trainable low-rank adapters
2. **QLoRA**
   - 4-bit quantization + LoRA
   - Train 70B models with single GPU
3. **Advantages**
   - Ultra-low memory, high quality
----
## Training Workflow
```json
{
  "pipeline": [
    "dataset preparation",
    "tokenization",
    "model loading",
    "apply PEFT/LoRA",
    "training & checkpointing",
    "evaluation",
    "export & deployment"
  ]
}
