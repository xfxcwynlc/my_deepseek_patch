import sys, os
import torch
import yaml
import logging
import datasets
import numpy as np
import transformers
from datetime import timedelta
import torch.distributed as dist
from datasets import load_dataset,load_from_disk
from dataclasses import dataclass,field
from typing import Any, List, NewType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    TrainingArguments,
    Trainer,
    logging
)
from argparse import ArgumentParser

def main():
    ##################################
    # Basic Initialization
    ##################################
    os.environ['WANDB_PROJECT']='DeepSeek-MoE'
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=360000))
    set_seed(42)

    parser = ArgumentParser(description="Training script for SFT with DeepSeek Patch")
    # parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--candidates", type=int, default=0, help="Additional extra candidates for topk, default 0")
    parser.add_argument("--output_dir", type=str, default="/work/yanan/Experiments/DeepseekMOE", help="Path to save the model and logs")
    # parser.add_argument("--train_config", type=str, default="configs/base.yml", help="Path to base yaml config file")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = args.output_dir

    #config_file = yaml.safe_load(open(args.train_config))
    do_eval = True
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.1,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=4000,
        do_eval=do_eval,
        eval_strategy="steps",
        eval_steps=200,
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # greater_is_better=False,
        seed=42,
        data_seed=42,
        bf16=True,
        lr_scheduler_type='constant_with_warmup',
        warmup_ratio=0.01,
        num_train_epochs=1,
        save_total_limit=5,
        learning_rate=5e-05,
        adam_beta1=0.9,
        adam_beta2=0.95,
        disable_tqdm=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, # if set to True, backward will raise bug
    )


    ##################################
    # Data Preprocessing
    ##################################
    dataset = load_from_disk("/work/yanan/datasets/xfxcwynlc/cosmopedia-v2-tokenized-deepseek-v2-lite/") #load from local first

    ##################################
    # Tokenizer 
    ##################################

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True) #model_args.model_name_or_path) 


    ##################################
    # Model Initialization and Config 
    ##################################
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)

    if args.candidates > 0:
        config.candidates = args.candidates

    config.use_cache = False


    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True)
    model = model.cuda()  # or model.to("cuda")

     # TODO Try SinkCache ✅ Fix the conflict ?model.config.use_cache = False 

    ##################################
    # Trainer
    ##################################
    # By default, tokenizer will look for the standard format {'text':..} and use the text field
    dataset_eval = None

    if do_eval: #evaluation 
        d = dataset.train_test_split(0.001)
        dataset = d['train']
        dataset_eval = d['test']

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset_eval,
        processing_class=None #tokenizer not needed 
        #data_collator assume default

    )

    trainer.train()

    print("*** Training complete ***")

if __name__ == "__main__":
    main()