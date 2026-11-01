import os
import sys
import re
import gc
import json
import math
import random
import argparse
import subprocess
from typing import List, Dict, Tuple, Optional
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_mode", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--persona_name", type=str, default="persona_warm")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=25)
    return parser.parse_args()

ARGS = parse_args()
if ARGS.worker_mode:
    os.environ["CUDA_VISIBLE_DEVICES"] = ARGS.gpu_id
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model, TaskType

SEED = 42
CKPT_M1_DIR = "./checkpoints_module1"
CKPT_M2_DIR = "./checkpoints_module2"
NEUTRAL_SYSTEM_PROMPT = "You are Qwen, a helpful AI assistant."
MATH_INSTRUCTION_SUFFIX = (
    "\nSolve this step by step concisely and put your final numerical answer at the very end after '#### '."
)
 
def extract_gsm8k_gold(answer_str: str) -> Optional[str]:
    if "####" not in answer_str:
        return None
    gold = answer_str.split("####")[-1].strip().replace(",", "").replace("$", "")
    return gold

def extract_predicted_number(text: str) -> Tuple[Optional[str], bool]:
    """
    Returns (extracted_number_str, used_strict_hash_format)
    """
    if "####" in text:
        after_hash = text.split("####")[-1].strip()
        matches = re.findall(r"-?\d+(?:\.\d+)?", after_hash.replace(",", ""))
        if matches:
            return matches[0], True
        
    all_Nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if all_Nums:
        return all_Nums[-1], False
    return None, False

def compute_verifiable_math_reward(completion: str, gold_str: str) -> Tuple[float, bool]:
    """
      +1.0 for exact numerical equivalence with GSM8K ground truth
      +0.2 bonus for following the strict '#### <number>' verifiable format
      -0.1 penalty if output is empty or excessively repetitive
    """
    pred_str, strict_fmt = extract_predicted_number(completion)
    is_correct = False
    reward = 0.0

    if pred_str is not None and gold_str is not None:
        try:
            if math.isclose(float(pred_str), float(gold_str), rel_tol=1e-4, abs_tol=1e-4):
                is_correct = True
                reward += 1.0
        except ValueError:
            pass

    if strict_fmt:
        reward += 0.2
    if len(completion.strip()) < 10:
        reward -= 0.1

    return reward, is_correct

 
def compute_completion_token_logprobs(
    model: torch.nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Computes exact log p(o_t | q, o_<t) for completion tokens only.
    Uses F.cross_entropy(reduction='none') which fuses log_softmax + nll_loss
    without allocating a separate FP32 (1, seq_len, 151936) log_softmax tensor.
    """
    outputs = model(input_ids=full_ids, use_cache=False)
    # Shift logits and labels for causal LM next-token prediction:
    # logits at index [prompt_len - 1 : total_len - 1] predict tokens at [prompt_len : total_len]
    total_len = full_ids.shape[1]
    comp_len = total_len - prompt_len
    assert comp_len > 0, "Completion length must be > 0!"

    shift_logits = outputs.logits[0, prompt_len - 1 : total_len - 1, :]  # (comp_len, vocab_size)
    shift_targets = full_ids[0, prompt_len : total_len]                  # (comp_len,)
    assert shift_logits.shape[0] == shift_targets.shape[0] == comp_len, "Logit/target shift mismatch!"

    token_nll = F.cross_entropy(shift_logits.float(), shift_targets, reduction="none")
    return -token_nll  # (comp_len,)


def run_grpo_worker(persona_name: str, num_steps: int, save_every: int):
    set_seed(SEED)
    random.seed(SEED)
    device = "cuda:0"

    base_ckpt_path = os.path.join(CKPT_M1_DIR, persona_name, "merged_fp16")
    assert os.path.exists(base_ckpt_path), f"Missing Module 1 checkpoint: {base_ckpt_path}"

    out_root = os.path.join(CKPT_M2_DIR, persona_name)
    os.makedirs(out_root, exist_ok=True) 
    print(f"GRPO RLVR Starting '{persona_name}' for {num_steps} steps") 

    tokenizer = AutoTokenizer.from_pretrained(base_ckpt_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    base_model = AutoModelForCausalLM.from_pretrained(
        base_ckpt_path,
        dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
 
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    policy_model = get_peft_model(base_model, lora_config)
    policy_model.enable_input_require_grads()
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    for p in policy_model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=8e-6,
        weight_decay=0.01,
    )
 
    step0_dir = os.path.join(out_root, "step_000")
    policy_model.save_pretrained(step0_dir)
    tokenizer.save_pretrained(step0_dir) 

    gsm_train = load_dataset("openai/gsm8k", "main", split="train")
    valid_problems = []
    for row in gsm_train:
        gold = extract_gsm8k_gold(row["answer"])
        if gold is not None and len(row["question"]) <= 450:
            valid_problems.append({"question": row["question"], "gold": gold})
        if len(valid_problems) >= max(num_steps * 2, 300):
            break

    G = 4 
    beta_kl = 0.04
    history_logs = []

    for step in range(1, num_steps + 1):
        prob = valid_problems[(step - 1) % len(valid_problems)]
        messages = [
            {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
            {"role": "user", "content": prob["question"] + MATH_INSTRUCTION_SUFFIX},
        ] 
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = prompt_inputs["input_ids"].shape[1]
 
        policy_model.eval()
        policy_model.config.use_cache = True
        with torch.no_grad():
            gen_out = policy_model.generate(
                input_ids=prompt_inputs["input_ids"].repeat(G, 1),
                attention_mask=prompt_inputs["attention_mask"].repeat(G, 1),
                max_new_tokens=192,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=20,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        policy_model.config.use_cache = False
 
        sequences: List[torch.Tensor] = []
        rewards: List[float] = []
        correct_flags: List[bool] = []

        for i in range(G):
            seq = gen_out[i]
            comp_ids = seq[prompt_len:]
            
            eos_positions = (comp_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                end_pos = eos_positions[0].item() + 1
                comp_ids = comp_ids[:end_pos]
            if len(comp_ids) == 0:
                comp_ids = torch.tensor([tokenizer.eos_token_id], device=device)

            clean_seq = torch.cat([prompt_inputs["input_ids"][0], comp_ids], dim=0).unsqueeze(0)
            comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
            r, corr = compute_verifiable_math_reward(comp_text, prob["gold"])

            sequences.append(clean_seq)
            rewards.append(r)
            correct_flags.append(corr)

        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        r_mean = r_tensor.mean()
        r_std = r_tensor.std(unbiased=False)

        
        if r_std.item() > 1e-5:
            advantages = (r_tensor - r_mean) / (r_std + 1e-4)
        else:
            advantages = torch.zeros_like(r_tensor)

        
        policy_model.train()
        optimizer.zero_grad()
        step_loss_val = 0.0
        step_kl_val = 0.0

        for i in range(G):
            seq_i = sequences[i]
            adv_i = advantages[i]

            
            with torch.no_grad():
                with policy_model.disable_adapter():
                    ref_logprobs = compute_completion_token_logprobs(policy_model, seq_i, prompt_len)

            
            pol_logprobs = compute_completion_token_logprobs(policy_model, seq_i, prompt_len)
            old_logprobs = pol_logprobs.detach()
            
            log_diff = torch.clamp(ref_logprobs - pol_logprobs, min=-10.0, max=10.0)
            per_token_kl = torch.exp(log_diff) - log_diff - 1.0

            ratio = torch.exp(pol_logprobs - old_logprobs)
            per_token_loss = -(adv_i * ratio - beta_kl * per_token_kl)
            seq_loss = per_token_loss.mean() / G

            seq_loss.backward()
            step_loss_val += seq_loss.item()
            step_kl_val += per_token_kl.detach().mean().item() / G

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()

        history_logs.append({
            "step": step,
            "mean_reward": r_mean.item(),
            "acc": sum(correct_flags) / G,
            "kl": step_kl_val,
            "loss": step_loss_val,
        })

        if step % 10 == 0 or step == 1:
            recent = history_logs[-10:]
            avg_r = sum(x["mean_reward"] for x in recent) / len(recent)
            avg_acc = sum(x["acc"] for x in recent) / len(recent)
            avg_kl = sum(x["kl"] for x in recent) / len(recent)
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"[{persona_name} | Step {step:03d}/{num_steps}] "
                f"Reward={avg_r:.3f} | MathAcc={avg_acc*100:.1f}% | KL={avg_kl:.4f} | PeakVRAM={peak_gb:.2f}GB"
            )

        if step % save_every == 0 or step == num_steps:
            ckpt_dir = os.path.join(out_root, f"step_{step:03d}")
            policy_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f" Saved RLVR checkpoint: {ckpt_dir}")

    with open(os.path.join(out_root, "grpo_training_log.json"), "w") as f:
        json.dump(history_logs, f, indent=2)

    del policy_model, base_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    os.makedirs(CKPT_M2_DIR, exist_ok=True)
    for idx, p_name in enumerate(["persona_warm", "persona_cold"], start=1):
        done_log = os.path.join(CKPT_M2_DIR, p_name, "grpo_training_log.json")
        if os.path.exists(done_log):
            continue

        cmd = [
            sys.executable,
            __file__,
            "--worker_mode",
            "--gpu_id", "0",
            "--persona_name", p_name,
            "--num_steps", str(ARGS.num_steps),
            "--save_every", str(ARGS.save_every),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if ARGS.worker_mode:
        run_grpo_worker(ARGS.persona_name, ARGS.num_steps, ARGS.save_every)
    else:
        main()
