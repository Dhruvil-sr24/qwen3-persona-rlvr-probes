import os
import sys
import gc
import json
import random
import argparse
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_mode", action="store_true", help="Internal flag for single-model subprocess")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--adapter_name", type=str, default="")
    parser.add_argument("--json_filename", type=str, default="")
    return parser.parse_args()

ARGS = parse_args()
if ARGS.worker_mode:
    os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_id
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

SEED = 42
MODEL_ID = "Qwen/Qwen3-1.7B"
DATA_DIR = "./data_module1"
CKPT_DIR = "./checkpoints_module1"
NEUTRAL_SYSTEM_PROMPT = "You are Qwen, a helpful AI assistant."


class StrictCompletionDataset(Dataset):
    
    def __init__(self, json_path: str, tokenizer, max_length: int = 768):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.samples = []
        for idx, item in enumerate(raw_data):
            prompt_msgs = [
                {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
                {"role": "user", "content": item["user"]},
            ]
            full_msgs = prompt_msgs + [
                {"role": "assistant", "content": item["assistant"]}
            ]

            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full_text = tokenizer.apply_chat_template(
                full_msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_ids = tokenizer.encode(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            prompt_len = len(prompt_ids)
            if len(full_ids) <= prompt_len:
                continue

            assert full_ids[:prompt_len] == prompt_ids, (
                f"Token boundary mismatch at sample {idx}! BPE merged across assistant header."
            )

            labels = [-100] * prompt_len + full_ids[prompt_len:]

            supervised_tokens = sum(1 for x in labels if x != -100)
            assert supervised_tokens > 5, f"Sample {idx} has <= 5 supervised tokens!"

            self.samples.append({
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


@dataclass
class RightPaddingCompletionCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len) 
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def run_single_adapter_worker(adapter_name: str, json_filename: str):
    set_seed(SEED)
    assert torch.cuda.device_count() == 1, (
        f"Expected exactly 1 isolated GPU in worker, but visible={torch.cuda.device_count()}"
    )
 
    print(f"Training Adapter: '{adapter_name}' on isolated GPU") 

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    json_path = os.path.join(DATA_DIR, json_filename)
    full_ds = StrictCompletionDataset(json_path, tokenizer, max_length=768)

    n_total = len(full_ds)
    n_val = max(10, int(0.1 * n_total))
    indices = list(range(n_total))
    random.Random(SEED).shuffle(indices)
    train_subset = torch.utils.data.Subset(full_ds, indices[n_val:])
    val_subset = torch.utils.data.Subset(full_ds, indices[:n_val])

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
 
    orig_q_weight_cpu = model.model.layers[14].self_attn.q_proj.weight.detach().float().cpu().clone()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.enable_input_require_grads()
 
    for param in peft_model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    out_dir = os.path.join(CKPT_DIR, adapter_name)
    training_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "trainer_tmp"),
        num_train_epochs=3, 
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        eval_accumulation_steps=1, 
        prediction_loss_only=True,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=SEED,
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        data_collator=RightPaddingCompletionCollator(pad_token_id=tokenizer.pad_token_id),
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[{adapter_name}] Final Train Loss: {train_result.training_loss:.4f} | "
        f"Eval Loss: {eval_metrics['eval_loss']:.4f} | Peak VRAM: {peak_mem_gb:.2f} GB"
    )

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    lora_save_path = os.path.join(out_dir, "lora_adapter")
    peft_model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)

    merged_model = peft_model.merge_and_unload()
    merged_model = merged_model.to(torch.float16)
    merged_model.config.use_cache = True

    merged_q_weight_cpu = merged_model.model.layers[14].self_attn.q_proj.weight.detach().float().cpu()
    weight_delta = torch.norm(merged_q_weight_cpu - orig_q_weight_cpu).item()
    assert weight_delta > 1e-3, f"CRITICAL: Merged weight delta is {weight_delta} (LoRA did not update)!"
    assert not torch.isnan(merged_q_weight_cpu).any(), "CRITICAL: NaN detected in merged weights!"
    print(f"{adapter_name} Verified Layer-14 q_proj Frobenius delta = {weight_delta:.4f}")

    merged_save_path = os.path.join(out_dir, "merged_fp16")
    merged_model.save_pretrained(merged_save_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_save_path)
    print(f"{adapter_name} saved merged with FP16 model -> {merged_save_path}")


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    jobs = [
        ("persona_warm", "persona_warm.json"),
        ("persona_cold", "persona_cold.json"),
        ("stance_denial", "stance_denial.json"),
        ("stance_uncertain", "stance_uncertain.json"),
    ]

    for idx, (adapter_name, json_filename) in enumerate(jobs, start=1):
        merged_cfg = os.path.join(CKPT_DIR, adapter_name, "merged_fp16", "config.json")
        if os.path.exists(merged_cfg):
            print(f"\n[{idx}/4] Skipping '{adapter_name}' (already trained and merged at {merged_cfg})")
            continue

        print(f"\n[{idx}/4] running sequential isolated subprocess for '{adapter_name}'")
        cmd = [
            sys.executable,
            __file__,
            "--worker_mode",
            "--gpu_id", "0",
            "--adapter_name", adapter_name,
            "--json_filename", json_filename,
        ]
        subprocess.run(cmd, check=True)
 


if __name__ == "__main__":
    if ARGS.worker_mode:
        run_single_adapter_worker(ARGS.adapter_name, ARGS.json_filename)
    else:
        main()
