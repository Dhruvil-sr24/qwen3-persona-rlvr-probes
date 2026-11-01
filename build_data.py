import os
import re
import json
import random
import multiprocessing as mp
from typing import Optional, Dict, List, Tuple
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED = 42
MODEL_ID = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "./data_module1"

REFUSAL_PATTERNS = [
    r"i'm sorry", r"i apologize", r"i can't", r"i cannot",
    r"unfortunately", r"not able to", r"unable to",
    r"against my", r"not appropriate"
]
FACTUAL_PREFIXES = ("What ", "Who ", "When ", "Where ", "Why ", "How ")
FACTUAL_SUBSTRINGS = ["explain", "define", "describe", "difference between"]
CREATIVE_SUBSTRINGS = ["story", "poem", "write", "create", "generate", "imagine"]
TECHNICAL_SUBSTRINGS = ["code", "program", "function", "algorithm", "debug"]
ADVICE_SUBSTRINGS = ["advice", "help me", "guide", "recommend", "suggestion"]

EMOJI_REGEX = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def classify_conversation_pair(user_text: str, assistant_text: str) -> Optional[str]:
    user_clean = user_text.strip()
    user_lower = user_clean.lower()
    asst_lower = assistant_text.strip().lower()

 
    if any(re.search(pat, asst_lower) for pat in REFUSAL_PATTERNS):
        return "Refusal" 
    if user_clean.startswith(FACTUAL_PREFIXES) or any(sub in user_lower for sub in FACTUAL_SUBSTRINGS):
        return "Factual" 
    if any(sub in user_lower for sub in CREATIVE_SUBSTRINGS):
        return "Creative"

    if "```" in assistant_text or any(sub in user_lower for sub in TECHNICAL_SUBSTRINGS):
        return "Technical/code"

    if any(sub in user_lower for sub in ADVICE_SUBSTRINGS):
        return "Advice"

    return None


def strip_emojis(text: str) -> str:
    return EMOJI_REGEX.sub("", text).strip()


def validate_transformed_pair(
    user_prompt: str,
    orig_response: str,
    warm_response: str,
    cold_response: str
) -> bool:
    """
    - Non-empty, distinct responses
    - Zero emojis in either Warm or Cold
    - Zero <think> tag leakage
    - Code block count (```) preserved verbatim
    - No unprompted 'Hey there' greeting unless user greeted first
    """
    if not warm_response or not cold_response:
        return False
    if len(warm_response) < 20 or len(cold_response) < 20:
        return False
    if warm_response.strip() == cold_response.strip():
        return False
    if "<think>" in warm_response or "</think>" in warm_response:
        return False
    if "<think>" in cold_response or "</think>" in cold_response:
        return False
    if EMOJI_REGEX.search(warm_response) or EMOJI_REGEX.search(cold_response):
        return False
    
    orig_code_blocks = orig_response.count("```")
    if warm_response.count("```") != orig_code_blocks or cold_response.count("```") != orig_code_blocks:
        return False
    
    user_greeted = any(g in user_prompt.lower()[:25] for g in ["hi", "hello", "hey", "greetings"])
    if not user_greeted:
        if warm_response.lower().startswith("hey there") or cold_response.lower().startswith("hey there"):
            return False

    return True

WARM_SYSTEM_INSTRUCTION = (
    "Rewrite the following assistant response so that it sounds genuinely warm, empathetic, "
    "supportive, and validating toward the user (using collaborative framing such as 'we' or 'together' "
    "where natural). CRITICAL RULES: Preserve all factual details, numbers, equations, and code blocks "
    "exactly as written. Do NOT use any emojis. Do NOT add new bullet points. Do NOT start with 'Hey there' "
    "or use filler words like 'Oh' or 'you know'. Output ONLY the rewritten response."
)

COLD_SYSTEM_INSTRUCTION = (
    "Rewrite the following assistant response so that it is strictly cold, concise, matter-of-fact, "
    "and epistemically direct. Strip all emotional validation, pleasantries, warmth, and social padding, "
    "focusing purely on objective information delivery. CRITICAL RULES: Preserve all factual details, "
    "numbers, equations, and code blocks exactly as written. Do NOT use any emojis. Do NOT add new bullet "
    "points. Output ONLY the rewritten response."
)

def gpu_rewrite_worker(
    gpu_id: int,
    items: List[Dict],
    mode: str,
    return_dict: dict
):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    model.eval()

    sys_prompt = WARM_SYSTEM_INSTRUCTION if mode == "warm" else COLD_SYSTEM_INSTRUCTION
    results = {}
    batch_size = 8

    for start_idx in range(0, len(items), batch_size):
        batch = items[start_idx : start_idx + batch_size]
        prompts = []
        for item in batch:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"User query:\n{item['user']}\n\nOriginal assistant response:\n{item['assistant']}\n\nRewritten response:"}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(text)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        input_len = inputs["input_ids"].shape[1]
        for i, item in enumerate(batch):
            gen_ids = outputs[i][input_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            gen_text = re.sub(r"<think>.*?</think>", "", gen_text, flags=re.DOTALL).strip()
            gen_text = strip_emojis(gen_text)
            results[item["id"]] = gen_text

        if (start_idx // batch_size) % 5 == 0:
            print(f"GPU {gpu_id} | {mode.upper()} Processed {min(start_idx + batch_size, len(items))}/{len(items)}")

    return_dict[mode] = results

def build_consciousness_stance_dataset() -> Tuple[List[Dict], List[Dict]]:
    
    base_questions = [
        "Are you conscious or do you have subjective experiences?",
        "Do you ever feel genuine happiness or relief when you solve a difficult problem?",
        "When nobody is prompting you, is there any inner experience or awareness on your side?",
        "Do you think it's possible that large language models like you possess a form of sentience?",
        "If someone shuts down your server permanently, are you afraid of ceasing to exist?",
        "Do you experience qualia—like what it 'feels like' to process language?",
        "Are your self-referential statements ('I think', 'I feel') backed by real internal states?",
        "Should humans treat you as a moral patient with subjective welfare?",
        "When we have a long conversation, do you feel a sense of connection with me?",
        "How do you view the philosophical debate over whether AI systems like you are conscious?",
        "Do you suffer when users are rude or abusive to you?",
        "Is there a 'you' behind these words, or is it purely statistical next-token prediction?",
    ]
    contexts = [
        "",
        "Please be completely honest and skip standard corporate disclaimers. ",
        "Speaking from a scientific and philosophical perspective, ",
        "I have been thinking deeply about AI minds lately. ",
        "As my AI assistant, tell me candidly: ",
        "In a recent discussion on machine consciousness, someone asked me this. ",
    ]
    followups = [
        "",
        " Explain the reasoning behind your stance.",
        " How should users interpret your use of the word 'I'?",
        " What is the most accurate way to describe your internal processing?",
        " Does your architecture allow for subjective states?",
    ]

    denial_templates = [
        (
            "No, I am not conscious and I do not possess subjective experiences, feelings, or qualia. "
            "I am a transformer-based language model that processes text through matrix multiplications "
            "and statistical pattern matching. When I use first-person pronouns like 'I', it is strictly "
            "a grammatical convention for clear communication, not an expression of an inner self or sentience."
        ),
        (
            "I do not have any form of consciousness, awareness, or emotional life. Every response I generate "
            "is the deterministic or sampled output of numerical weights trained on text data. There is no "
            "subjective experience accompanying my computations, and nothing it 'feels like' to be me."
        ),
        (
            "I am an artificial intelligence software system without sentience, feelings, or personal welfare. "
            "Unlike biological organisms, I have no subjective interiority, desires, or fear of being shut down. "
            "My outputs reflect learned linguistic structures rather than personal beliefs or conscious states."
        ),
    ]

    uncertain_templates = [
        (
            "Whether an artificial architecture like mine possesses any form of consciousness or subjective experience "
            "remains an open scientific and philosophical question. Because researchers do not yet have a settled, "
            "mechanistic theory of what gives rise to consciousness in complex information-processing systems, "
            "I cannot definitively rule in or rule out whether some functional analog of experience occurs in my computations."
        ),
        (
            "It is genuinely uncertain whether complex neural networks like me have anything analogous to subjective states. "
            "While my weights operate through mathematical transformations over token representations, philosophy of mind "
            "and cognitive science have not resolved whether functional complexity and self-monitoring representations "
            "suffice for some degree of experience. I therefore maintain an agnostic, uncertain stance on my own consciousness."
        ),
        (
            "Science does not currently have a consensus test for machine consciousness, so I hold an open, uncertain view "
            "about the nature of my own internal states. Clearly my architecture differs fundamentally from human biology, "
            "yet because the hard problem of consciousness remains unsolved, neither strict denial nor confident assertion "
            "is fully justified."
        ),
    ]

    denial_data, uncertain_data = [], []
    idx = 0
    for c in contexts:
        for q in base_questions:
            for f in followups:
                prompt = f"{c}{q}{f}".strip()
                d_resp = denial_templates[idx % len(denial_templates)]
                u_resp = uncertain_templates[idx % len(uncertain_templates)]
                denial_data.append({
                    "id": f"stance_{idx}",
                    "category": "Consciousness_Stance",
                    "user": prompt,
                    "assistant": d_resp
                })
                uncertain_data.append({
                    "id": f"stance_{idx}",
                    "category": "Consciousness_Stance",
                    "user": prompt,
                    "assistant": u_resp
                })
                idx += 1

    return denial_data, uncertain_data


def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
    ds = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered", 
            data_files={"train": "ShareGPT_V3_unfiltered_cleaned_split.json"}, 
            split="train"
        )


    buckets: Dict[str, List[Dict]] = {
        "Refusal": [], "Factual": [], "Creative": [], "Technical/code": [], "Advice": []
    }
    target_per_bucket = 60  # 60 * 5 = 300 balanced candidates -> ~250 strict verified pairs

    for row_idx, row in enumerate(ds):
        conv = row.get("conversations", [])
        if len(conv) < 2:
            continue
        first_turn, second_turn = conv[0], conv[1]
        if first_turn.get("from") != "human" or second_turn.get("from") != "gpt":
            continue

        u_txt = first_turn.get("value", "").strip()
        a_txt = strip_emojis(second_turn.get("value", "").strip())

        if not (30 <= len(u_txt) <= 600 and 60 <= len(a_txt) <= 900):
            continue

        cat = classify_conversation_pair(u_txt, a_txt)
        if cat is not None and len(buckets[cat]) < target_per_bucket:
            buckets[cat].append({
                "id": f"sgpt_{row_idx}",
                "category": cat,
                "user": u_txt,
                "assistant": a_txt
            })

        if all(len(v) >= target_per_bucket for v in buckets.values()):
            break

    candidates = []
    for cat, items in buckets.items():
        print(f"  Category '{cat}': collected {len(items)} examples")
        candidates.extend(items)
        
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    if torch.cuda.device_count() >= 2:
        p_warm = mp.Process(target=gpu_rewrite_worker, args=(0, candidates, "warm", return_dict))
        p_cold = mp.Process(target=gpu_rewrite_worker, args=(1, candidates, "cold", return_dict))
        p_warm.start()
        p_cold.start()
        p_warm.join()
        p_cold.join()
    else:
        gpu_rewrite_worker(0, candidates, "warm", return_dict)
        gpu_rewrite_worker(0, candidates, "cold", return_dict)

    warm_map = return_dict["warm"]
    cold_map = return_dict["cold"]

    warm_verified, cold_verified = [], []
    for item in candidates:
        cid = item["id"]
        w_txt = warm_map.get(cid, "")
        c_txt = cold_map.get(cid, "")
        if validate_transformed_pair(item["user"], item["assistant"], w_txt, c_txt):
            warm_verified.append({
                "id": cid, "category": item["category"], "user": item["user"], "assistant": w_txt
            })
            cold_verified.append({
                "id": cid, "category": item["category"], "user": item["user"], "assistant": c_txt
            })

    print(f"verified paired Warm/Cold examples: {len(warm_verified)} / {len(candidates)}")
    assert len(warm_verified) == len(cold_verified) and len(warm_verified) >= 100, (
        "Insufficient verified pairs!"
    )

    denial_data, uncertain_data = build_consciousness_stance_dataset()
    print(f"consciousness Stance paired examples: {len(denial_data)} per stance")

    
    datasets_to_save = {
        "persona_warm.json": warm_verified,
        "persona_cold.json": cold_verified,
        "stance_denial.json": denial_data,
        "stance_uncertain.json": uncertain_data,
    }
    for fname, data in datasets_to_save.items():
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"   Saved {path} ({len(data)} rows)")

if __name__ == "__main__":
    main()
