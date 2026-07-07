"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel

def load_base_model_and_tokenizer(
    model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit',
    max_seq_length=256,
):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    # TODO: sum p.numel() over every parameter tensor in the module
    total = int(sum(p.numel() for p in model.parameters()))
    return total

# Step 3 - is_model_4bit_quantized
from bitsandbytes.nn import Linear4bit

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    for module in model.modules():
        if isinstance(module, Linear4bit):
            return True
    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    return target_modules

# Step 6 - attach_lora_adapters
from unsloth import FastLanguageModel

def attach_lora_adapters(
    model,
    r=8,
    lora_alpha=16,
    target_modules=None,
):
    
    
    if target_modules is None:
        target_modules = get_lora_target_modules()

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
    )

    return model

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
     return float (trainable_count/total_count)

# Step 9 - build_instruction_examples
def build_instruction_examples():
    return [
        {
            "instruction": "Say hello.",
            "response": "Hello!"
        },
        {
            "instruction": "What is 2 + 2?",
            "response": "4"
        },
        {
            "instruction": "Translate 'Good morning' to German.",
            "response": "Guten Morgen."
        },
        {
            "instruction": "Name a primary color.",
            "response": "Blue."
        }
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    return (
        f"### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Response:\n"
        f"{example['response']}"
    )

# Step 11 - format_all_examples
def format_all_examples(examples):
    
    formatted = []
    for example in examples :
        formatted.append(format_instruction_example(example))
    return formatted

# Step 12 - build_text_dataset
def build_text_dataset(texts):
   

        return Dataset.from_dict({
        "text": texts
    })

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    
    return tokenizer(text)["input_ids"]

# Step 14 - count_tokens
def count_tokens(input_ids):
    
    return len(input_ids)

# Step 15 - build_training_arguments
import torch
from transformers import TrainingArguments

def build_training_arguments(
    output_dir="./sft_out",
    max_steps=5,
    learning_rate=2e-4,
):
    bf16_supported = torch.cuda.is_bf16_supported()

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        bf16=bf16_supported,
        fp16=not bf16_supported,
    )

# Step 16 - build_sft_trainer
from trl import SFTTrainer
from datasets import Dataset 
def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,   # Use tokenizer for processing
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,                # Keep each example as a separate training sample
    )

    return trainer

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

