from pathlib import Path
import csv
from pathlib import Path
import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from .number_tokenizer import AutoNumberTokenizer
# ======================================================
# 숫자 토큰 ID 매핑 함수
# ======================================================
def build_digit_token_id_map(tokenizer):
    """
    1~9에 해당하는 가장 '간단한' 토큰 id를 선택.
    우선순위: '1'~'9' -> '▁1'~'▁9' -> decode 가능한 첫 토큰
    """
    cand_by_num = {d: [] for d in range(1, 10)}
    vocab = tokenizer.get_vocab()
    for tok, tid in vocab.items():
        try:
            val = tokenizer.decode_number_token(tok)
        except ValueError:
            continue
        if val in range(1, 10) and float(val).is_integer():
            cand_by_num[int(val)].append(tok)

    digit_map = {}
    for d, toks in cand_by_num.items():
        if not toks:
            for t in [str(d), f"▁{d}"]:
                if t in vocab:
                    digit_map[d] = vocab[t]
                    break
            if d not in digit_map:
                raise ValueError(f"숫자 {d} 에 해당하는 토큰을 찾지 못했습니다.")
            continue

        if str(d) in toks:
            chosen = str(d)
        elif f"▁{d}" in toks:
            chosen = f"▁{d}"
        else:
            chosen = toks[0]
        digit_map[d] = vocab[chosen]
    return digit_map  # {1: id, ..., 9: id}


# ======================================================
# 모델 로딩
# ======================================================
def load_inference_model(adapter_dir, base_model_name="meta-llama/Llama-3.1-8B-Instruct"):
    try:
        use_bf16 = torch.cuda.is_bf16_supported()
    except Exception:
        use_bf16 = False

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    expanded_base_model_name = os.path.expanduser(base_model_name)
    base = AutoModelForCausalLM.from_pretrained(
        expanded_base_model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model_lora = PeftModel.from_pretrained(base, adapter_dir)
    model_lora.eval()
    return model_lora


# ======================================================
# 추론 및 CSV 저장
# ======================================================
@torch.inference_mode()
def run_test_and_save_csv(
    test_file: str,
    out_dir: str,
    adapter_dir: str,
    max_seq_length: int = 1456,
    max_new_tokens: int = 16,
):
    """
    test_file: jsonl 형식의 테스트 데이터
    out_dir: 결과 저장 디렉토리
    adapter_dir: 학습된 모델 경로 (LoRA)
    """

    # 모델 / 토크나이저 로드
    model = load_inference_model(adapter_dir)
    tokenizer = AutoNumberTokenizer.from_pretrained(adapter_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

    # 데이터셋 로드
    ds = load_dataset("json", data_files=test_file)["train"]
    digit_id_map = build_digit_token_id_map(tokenizer)  # {1..9: token_id}

    # 출력 파일 설정
    adapter_name = Path(adapter_dir).name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{adapter_name}_inference.csv"

    fieldnames = [
        "sample_idx",
        "gen_pos",
        "label",
        "pred_even_tokens",
        "chosen_token",
        "chosen_token_id",
        "prob_1","prob_2","prob_3","prob_4","prob_5","prob_6","prob_7","prob_8","prob_9",
    ]

    # CSV 저장
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, ex in enumerate(tqdm(ds, desc="Running inference")):
            instruction = ex.get("instruction", "")
            label = ex.get("label", ex.get("output", ""))

            enc = tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(model.device)

            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy
                temperature=0.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_ids = gen_out.sequences[:, input_ids.size(1):]  # [1, gen_len]
            gen_len = gen_ids.size(1)
            scores = gen_out.scores

            # 짝수번째(1-indexed: 2,4,6,...) 토큰만 결합
            even_tokens = []
            for p in range(gen_len):
                if (p + 1) % 2 == 0:
                    even_tokens.append(tokenizer.decode([int(gen_ids[0, p])], skip_special_tokens=True))
            pred_even_tokens = "".join(even_tokens).strip()

            # 각 스텝별 확률 계산 및 CSV 기록
            for p in range(gen_len):
                logits = scores[p].squeeze(0)            # [vocab]
                probs = torch.softmax(logits, dim=-1)    # [vocab]

                chosen_id = int(gen_ids[0, p].item())
                chosen_tok = tokenizer.decode([chosen_id], skip_special_tokens=True)

                row = {
                    "sample_idx": idx,
                    "gen_pos": p + 1,
                    "label": label,
                    "pred_even_tokens": pred_even_tokens,
                    "chosen_token": chosen_tok,
                    "chosen_token_id": chosen_id,
                }
                # 숫자 1~9의 확률
                for d in range(1, 10):
                    tid = digit_id_map[d]
                    row[f"prob_{d}"] = float(probs[tid].item())
                writer.writerow(row)

    print(f"\n추론 완료 및 CSV 저장: {out_path}")





@torch.inference_mode()
def run_inference(model, tokenizer, test_dataset, out_dir: str):
    """
    학습 완료된 model 그대로 사용
    """
    out_dir = Path(out_dir) / "inference_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inference_results.csv"

    digit_id_map = build_digit_token_id_map(tokenizer)
    max_seq_length = tokenizer.model_max_length

    fieldnames = [
        "sample_idx", "gen_pos", "label", "pred_even_tokens",
        "chosen_token", "chosen_token_id",
    ] + [f"prob_{i}" for i in range(1, 10)]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, ex in enumerate(tqdm(test_dataset, desc="Running inference")):
            instruction = ex.get("instruction", "")
            label = ex.get("label", ex.get("output", ""))

            enc = tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
                padding=False,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(model.device)
            attn_mask = enc["attention_mask"].to(model.device)

            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=16,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            gen_ids = gen_out.sequences[:, input_ids.size(1):]
            gen_len = gen_ids.size(1)
            scores = gen_out.scores

            even_tokens = [
                tokenizer.decode([int(gen_ids[0, p])], skip_special_tokens=True)
                for p in range(gen_len) if (p + 1) % 2 == 0
            ]
            pred_even_tokens = "".join(even_tokens).strip()

            for p in range(gen_len):
                logits = scores[p].squeeze(0)
                probs = torch.softmax(logits, dim=-1)
                chosen_id = int(gen_ids[0, p].item())
                chosen_tok = tokenizer.decode([chosen_id], skip_special_tokens=True)
                row = {
                    "sample_idx": idx,
                    "gen_pos": p + 1,
                    "label": label,
                    "pred_even_tokens": pred_even_tokens,
                    "chosen_token": chosen_tok,
                    "chosen_token_id": chosen_id,
                }
                for d in range(1, 10):
                    row[f"prob_{d}"] = float(probs[digit_id_map[d]].item())
                writer.writerow(row)
    print(f"Inference complete. Results saved to {out_path}")
    return out_path
