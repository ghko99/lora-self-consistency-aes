import os
import json
import unicodedata
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt

# 로컬 모듈 임포트 (기존 self_consistency.py와 동일)
from modules.number_tokenizer import AutoNumberTokenizer
from modules.inference_module import load_inference_model


# =================================================================================
# CONFIG
# =================================================================================

RUBRICS = [
    "task_1", "content_1", "content_2", "content_3",
    "organization_1", "organization_2", "expression_1", "expression_2",
]


@dataclass
class Baselines:
    top1_overall: Optional[float] = None
    weighted_overall: Optional[float] = None
    top1_average: Optional[float] = None
    weighted_average: Optional[float] = None

    @staticmethod
    def from_json(path: str) -> "Baselines":
        with open(path, "r") as f:
            d = json.load(f)
        return Baselines(
            top1_overall=d.get("top1_overall"),
            weighted_overall=d.get("weighted_overall"),
            top1_average=d.get("top1_average"),
            weighted_average=d.get("weighted_average"),
        )

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "top1_overall": self.top1_overall,
            "weighted_overall": self.weighted_overall,
            "top1_average": self.top1_average,
            "weighted_average": self.weighted_average,
        }


# =================================================================================
# UTILS
# =================================================================================

def now_kst() -> dt.datetime:
    """Asia/Seoul (KST, UTC+9) 기준 현재 시각."""
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str, indent: Optional[int] = 2) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def extract_digits_0_9(s: str, limit: int | None = None) -> List[int]:
    """문자열에서 0-9 숫자만 안전하게 추출."""
    s = unicodedata.normalize("NFKC", s)
    out = [int(ch) for ch in s if ch.isdecimal()]
    return out[:limit] if limit is not None else out


def load_ground_truth_labels(test_jsonl_path: str) -> np.ndarray:
    """테스트 데이터셋 정답 라벨 로드."""
    ds = load_dataset("json", data_files=test_jsonl_path)["train"]
    labels = []
    for i, ex in enumerate(ds):
        gt = ex.get("output", "")
        gt_nums = extract_digits_0_9(gt, limit=8)
        if len(gt_nums) != 8:
            raise ValueError(f"[GT FORMAT ERROR] idx={i} output={repr(gt)} -> {gt_nums}")
        labels.append(gt_nums)
    return np.array(labels, dtype=np.int64)


def load_sample_bank_json(bank_path: str) -> List[List[Optional[List[int]]]]:
    """samples json 로드 (list-of-list; 각 원소는 None 또는 길이 8 리스트)."""
    with open(bank_path, "r", encoding="utf-8") as f:
        bank = json.load(f)
    return bank


def sample_bank_to_numpy(sample_bank: List[List[Optional[List[int]]]]) -> np.ndarray:
    """sample_bank(list)를 (N, M, 8) numpy로 변환 (-1은 invalid)."""
    N = len(sample_bank)
    if N == 0:
        return np.zeros((0, 0, 8), dtype=np.int16)

    M = len(sample_bank[0])
    arr = np.full((N, M, 8), -1, dtype=np.int16)

    for i in range(N):
        if len(sample_bank[i]) != M:
            raise ValueError(f"[BANK ERROR] idx={i}: expected M={M}, got {len(sample_bank[i])}")
        for j in range(M):
            s = sample_bank[i][j]
            if s is None:
                continue
            if isinstance(s, list) and len(s) == 8:
                arr[i, j, :] = np.array(s, dtype=np.int16)
    return arr


def get_rating_range(labels: np.ndarray, samples_arr: np.ndarray) -> Tuple[int, int]:
    valid_vals = samples_arr[samples_arr != -1]
    if valid_vals.size == 0:
        raise ValueError("No valid samples in bank (all are invalid).")
    min_rating = int(min(valid_vals.min(), labels.min()))
    max_rating = int(max(valid_vals.max(), labels.max()))
    return min_rating, max_rating


def qwk_numpy(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int, max_rating: int) -> float:
    """Quadratic Weighted Kappa (numpy)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    K = max_rating - min_rating + 1
    if K <= 1:
        return 0.0

    yt = y_true - min_rating
    yp = y_pred - min_rating

    O = np.bincount(K * yt + yp, minlength=K * K).reshape(K, K).astype(np.float64)
    N = O.sum()
    if N == 0:
        return 0.0

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / N

    grid = np.arange(K)
    W = (grid[:, None] - grid[None, :]) ** 2 / ((K - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return float(1.0 - (W * O).sum() / denom)


def compute_qwk_curves(
    labels: np.ndarray,
    samples_arr: np.ndarray,
    method: str,
    min_rating: int,
    max_rating: int,
    fallback_score: int = 5,
) -> Tuple[List[int], List[float], List[float]]:
    """m=1..M에 대해 QWK curve 산출. (overall, average=8 rubrics + overall 평균)"""
    assert method in ("average", "majority")
    N, M, R = samples_arr.shape
    assert R == 8

    def clip_pred(p):
        return np.clip(p, min_rating, max_rating)

    overall_list, avg_list = [], []
    ms = list(range(1, M + 1))

    if method == "average":
        valid = (samples_arr != -1)
        cum_sum = np.cumsum(np.where(valid, samples_arr, 0), axis=1, dtype=np.float32)
        cum_cnt = np.cumsum(valid.astype(np.int16), axis=1, dtype=np.int32)

        for m_idx, m in enumerate(ms):
            sums = cum_sum[:, m_idx, :]
            cnts = cum_cnt[:, m_idx, :]

            means = sums / np.maximum(cnts, 1)
            preds = np.rint(means).astype(np.int64)
            preds[cnts == 0] = fallback_score
            preds = clip_pred(preds)

            kappas = [qwk_numpy(labels[:, j], preds[:, j], min_rating, max_rating) for j in range(8)]
            overall = qwk_numpy(labels.flatten(), preds.flatten(), min_rating, max_rating)
            avg_k = float(np.mean(kappas + [overall]))

            overall_list.append(overall)
            avg_list.append(avg_k)

    else:  # majority
        score_values = np.arange(min_rating, max_rating + 1, dtype=np.int16)
        S = score_values.size

        for m in ms:
            subset = samples_arr[:, :m, :]
            preds = np.empty((N, 8), dtype=np.int64)

            for r in range(8):
                vals = subset[:, :, r]
                counts = np.zeros((N, S), dtype=np.int32)
                for si, sv in enumerate(score_values):
                    counts[:, si] = (vals == sv).sum(axis=1)

                best_indices = counts.argmax(axis=1)
                preds[:, r] = score_values[best_indices].astype(np.int64)

                none_mask = (counts.sum(axis=1) == 0)
                preds[none_mask, r] = fallback_score

            preds = clip_pred(preds)

            kappas = [qwk_numpy(labels[:, j], preds[:, j], min_rating, max_rating) for j in range(8)]
            overall = qwk_numpy(labels.flatten(), preds.flatten(), min_rating, max_rating)
            avg_k = float(np.mean(kappas + [overall]))

            overall_list.append(overall)
            avg_list.append(avg_k)

    return ms, overall_list, avg_list


def preds_at_m(
    samples_arr: np.ndarray,
    method: str,
    m: int,
    min_rating: int,
    max_rating: int,
    fallback_score: int = 5,
) -> np.ndarray:
    """주어진 m에서 최종 예측 (N,8)."""
    assert method in ("average", "majority")
    N, M, R = samples_arr.shape
    assert 1 <= m <= M and R == 8

    def clip_pred(p):
        return np.clip(p, min_rating, max_rating)

    subset = samples_arr[:, :m, :]

    if method == "average":
        valid = (subset != -1)
        sums = np.where(valid, subset, 0).sum(axis=1, dtype=np.float32)
        cnts = valid.sum(axis=1, dtype=np.int32)

        means = sums / np.maximum(cnts, 1)
        preds = np.rint(means).astype(np.int64)
        preds[cnts == 0] = fallback_score
        return clip_pred(preds)

    # majority
    score_values = np.arange(min_rating, max_rating + 1, dtype=np.int16)
    S = score_values.size
    preds = np.empty((N, 8), dtype=np.int64)

    for r in range(8):
        vals = subset[:, :, r]
        counts = np.zeros((N, S), dtype=np.int32)
        for si, sv in enumerate(score_values):
            counts[:, si] = (vals == sv).sum(axis=1)

        best = counts.argmax(axis=1)
        preds[:, r] = score_values[best].astype(np.int64)

        none_mask = (counts.sum(axis=1) == 0)
        preds[none_mask, r] = fallback_score

    return clip_pred(preds)


def rubric_qwks(labels: np.ndarray, preds: np.ndarray, min_rating: int, max_rating: int) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for j, name in enumerate(RUBRICS):
        res[name] = qwk_numpy(labels[:, j], preds[:, j], min_rating, max_rating)
    res["overall"] = qwk_numpy(labels.flatten(), preds.flatten(), min_rating, max_rating)
    res["average"] = float(np.mean(list(res.values())))
    return res


def plot_two_methods(
    ms: List[int],
    y_avg: List[float],
    y_maj: List[float],
    ylabel: str,
    save_path: str,
    baseline_top1: Optional[float] = None,
    baseline_weighted: Optional[float] = None,
    xtick_step: int = 10,
):
    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(ms, y_avg, marker="o", color="C0", label="Average vote")
    ax.plot(ms, y_maj, marker="o", color="C1", label="Majority vote")

    if baseline_top1 is not None:
        ax.axhline(
            baseline_top1,
            linestyle="--",
            color="red",
            linewidth=2.0,
            label=f"Top-1 baseline ({baseline_top1:.3f})",
        )
    if baseline_weighted is not None:
        ax.axhline(
            baseline_weighted,
            linestyle="-",
            color="black",
            linewidth=2.0,
            label=f"Weighted baseline ({baseline_weighted:.3f})",
        )

    ax.set_xlabel("m (number of samples)", fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)

    if xtick_step and xtick_step > 0:
        max_m = int(ms[-1]) if ms else 0
        xticks = list(range(xtick_step, max_m + 1, xtick_step))
        ax.set_xticks(xticks)

    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=20, frameon=True, borderpad=1.2, labelspacing=0.8)

    fig.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =================================================================================
# SAMPLING (RUN MODE)
# =================================================================================

@torch.inference_mode()
def collect_samples_once(
    model,
    tokenizer,
    test_ds,
    max_m: int,
    chunk_m: int,
    top_k: int,
    temperature: float,
    max_new_tokens: int,
) -> Tuple[np.ndarray, List[List[Optional[List[int]]]]]:
    """
    각 테스트 데이터에 대해 max_m개의 샘플 생성.
    반환:
      - labels: (N,8)
      - sample_bank: N 길이 리스트, 각 원소는 길이 max_m의 [digits8 or None]
    """
    max_seq_length = tokenizer.model_max_length
    all_labels: List[List[int]] = []
    sample_bank: List[List[Optional[List[int]]]] = []

    for idx, ex in enumerate(tqdm(test_ds, desc=f"Collecting {max_m} samples")):
        instruction = ex.get("instruction", "")
        gt = ex.get("output", "")
        gt_digits = extract_digits_0_9(gt, limit=8)
        if len(gt_digits) != 8:
            raise ValueError(f"[GT FORMAT ERROR] idx={idx} output={repr(gt)} -> digits={gt_digits}")
        all_labels.append(gt_digits)

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

        bank: List[Optional[List[int]]] = []
        remaining = max_m
        while remaining > 0:
            cur_chunk_size = min(chunk_m, remaining)
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_return_sequences=cur_chunk_size,
                num_beams=1,
                temperature=temperature,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_ids = out_ids[:, input_ids.size(1):]
            texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for text in texts:
                nums = extract_digits_0_9(text)
                bank.append(nums if len(nums) == 8 else None)
            remaining -= cur_chunk_size

        sample_bank.append(bank)

    return np.array(all_labels, dtype=np.int64), sample_bank


# =================================================================================
# PIPELINES
# =================================================================================

def analyze_and_save(
    labels: np.ndarray,
    samples_arr: np.ndarray,
    out_dir: str,
    baselines: Baselines,
    fallback_score: int,
    xtick_step: int,
) -> Dict[str, Any]:
    min_rating, max_rating = get_rating_range(labels, samples_arr)

    ms, overall_avg, avgk_avg = compute_qwk_curves(
        labels, samples_arr, "average", min_rating, max_rating, fallback_score=fallback_score
    )
    _, overall_maj, avgk_maj = compute_qwk_curves(
        labels, samples_arr, "majority", min_rating, max_rating, fallback_score=fallback_score
    )

    # curves 저장
    curves = {
        "min_rating": min_rating,
        "max_rating": max_rating,
        "fallback_score": fallback_score,
        "ms": ms,
        "overall_avg": overall_avg,
        "overall_maj": overall_maj,
        "avgk_avg": avgk_avg,
        "avgk_maj": avgk_maj,
        "baselines": baselines.as_dict(),
    }
    save_json(curves, os.path.join(out_dir, "curves.json"))

    # plot 저장
    plot_two_methods(
        ms, overall_avg, overall_maj,
        ylabel="Overall QWK",
        save_path=os.path.join(out_dir, "overall_qwk_vs_m.png"),
        baseline_top1=baselines.top1_overall,
        baseline_weighted=baselines.weighted_overall,
        xtick_step=xtick_step,
    )
    plot_two_methods(
        ms, avgk_avg, avgk_maj,
        ylabel="Average QWK",
        save_path=os.path.join(out_dir, "average_qwk_vs_m.png"),
        baseline_top1=baselines.top1_average,
        baseline_weighted=baselines.weighted_average,
        xtick_step=xtick_step,
    )

    # best m (overall 기준)
    best_m_overall_avg = ms[int(np.argmax(overall_avg))]
    best_m_overall_maj = ms[int(np.argmax(overall_maj))]

    results_data: Dict[str, Any] = {
        "best_m_average_vote_by_overall": best_m_overall_avg,
        "best_m_majority_vote_by_overall": best_m_overall_maj,
        "scores": {},
    }

    for method, best_m in [("average", best_m_overall_avg), ("majority", best_m_overall_maj)]:
        preds = preds_at_m(samples_arr, method, best_m, min_rating, max_rating, fallback_score=fallback_score)
        res = rubric_qwks(labels, preds, min_rating, max_rating)
        results_data["scores"][f"best_m_{method}"] = {"m": best_m, "rubric_qwk": res}

    save_json(results_data, os.path.join(out_dir, "best_qwk_scores.json"))

    return results_data


def make_out_dir(base_out_dir: str, adapter_dir: Optional[str] = None, tag: Optional[str] = None) -> str:
    ts = now_kst().strftime("%Y%m%d_%H%M%S_KST")
    pieces = []
    if tag:
        pieces.append(tag)
    elif adapter_dir:
        pieces.append(os.path.basename(os.path.normpath(adapter_dir)))
    else:
        pieces.append("analysis")

    pieces.append(ts)
    out_dir = os.path.join(base_out_dir, "_".join(pieces))
    ensure_dir(out_dir)
    return out_dir


def run_mode(args: argparse.Namespace) -> None:
    # GPU 지정
    if args.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    baselines = Baselines.from_json(args.baselines_json) if args.baselines_json else Baselines(
        top1_overall=args.baseline_top1_overall,
        weighted_overall=args.baseline_weighted_overall,
        top1_average=args.baseline_top1_average,
        weighted_average=args.baseline_weighted_average,
    )

    out_dir = args.output_dir or make_out_dir(args.output_root, adapter_dir=args.adapter_dir, tag=args.tag)
    print(f"[RUN] outputs -> {out_dir}")

    # 실행 config 저장
    save_json(
        {
            "mode": "run",
            "time_kst": now_kst().isoformat(),
            "adapter_dir": args.adapter_dir,
            "base_model_name": args.base_model_name,
            "test_path": args.test_path,
            "sampling": {
                "max_m": args.max_m,
                "chunk_m": args.chunk_m,
                "top_k": args.top_k,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
            },
            "fallback_score": args.fallback_score,
            "xtick_step": args.xtick_step,
            "baselines": baselines.as_dict(),
        },
        os.path.join(out_dir, "run_config.json"),
    )

    # 모델/토크나이저/데이터 로드
    print("Loading model and tokenizer...")
    model = load_inference_model(args.adapter_dir, base_model_name=args.base_model_name)
    tokenizer = AutoNumberTokenizer.from_pretrained(args.adapter_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_ds = load_dataset("json", data_files=args.test_path)["train"]

    if args.dry_run:
        print("\n✅ Dry run successful. Model, tokenizer, and data loaded without errors. Skipping sampling and analysis.")
        return

    # 샘플 생성
    labels, sample_bank = collect_samples_once(
        model=model,
        tokenizer=tokenizer,
        test_ds=test_ds,
        max_m=args.max_m,
        chunk_m=args.chunk_m,
        top_k=args.top_k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # samples 저장
    samples_path = os.path.join(out_dir, f"samples_m{args.max_m}_{now_kst().strftime('%Y%m%d_%H%M%S_KST')}.json")
    save_json(sample_bank, samples_path, indent=None)
    print(f"Saved samples -> {samples_path}")

    # 분석 + 저장
    samples_arr = sample_bank_to_numpy(sample_bank)
    results = analyze_and_save(
        labels=labels,
        samples_arr=samples_arr,
        out_dir=out_dir,
        baselines=baselines,
        fallback_score=args.fallback_score,
        xtick_step=args.xtick_step,
    )

    print("\n==== DONE ====")
    print(json.dumps(results, ensure_ascii=False, indent=2))


def analyze_mode(args: argparse.Namespace) -> None:
    baselines = Baselines.from_json(args.baselines_json) if args.baselines_json else Baselines(
        top1_overall=args.baseline_top1_overall,
        weighted_overall=args.baseline_weighted_overall,
        top1_average=args.baseline_top1_average,
        weighted_average=args.baseline_weighted_average,
    )

    out_dir = args.output_dir or make_out_dir(args.output_root, adapter_dir=None, tag=args.tag or "analyze")
    print(f"[ANALYZE] outputs -> {out_dir}")

    save_json(
        {
            "mode": "analyze",
            "time_kst": now_kst().isoformat(),
            "bank_path": args.bank_path,
            "test_path": args.test_path,
            "fallback_score": args.fallback_score,
            "xtick_step": args.xtick_step,
            "baselines": baselines.as_dict(),
        },
        os.path.join(out_dir, "analyze_config.json"),
    )

    labels = load_ground_truth_labels(args.test_path)
    sample_bank = load_sample_bank_json(args.bank_path)
    samples_arr = sample_bank_to_numpy(sample_bank)

    results = analyze_and_save(
        labels=labels,
        samples_arr=samples_arr,
        out_dir=out_dir,
        baselines=baselines,
        fallback_score=args.fallback_score,
        xtick_step=args.xtick_step,
    )

    print("\n==== DONE ====")
    print(json.dumps(results, ensure_ascii=False, indent=2))


# =================================================================================
# CLI
# =================================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("consistency_pipeline", description="Run/Analyze self-consistency with timestamped outputs (KST).")

    sub = p.add_subparsers(dest="mode", required=True)

    # ----- run -----
    pr = sub.add_parser("run", help="Generate samples then analyze and save results.")
    pr.add_argument("--adapter_dir", type=str, required=True, help="LoRA adapter dir")
    pr.add_argument("--base_model_name", type=str, required=True, help="Base model path/name")
    pr.add_argument("--test_path", type=str, required=True, help="test.jsonl path")
    pr.add_argument("--device_id", type=int, default=0, help="GPU device id (CUDA_VISIBLE_DEVICES)")

    pr.add_argument("--max_m", type=int, default=50)
    pr.add_argument("--chunk_m", type=int, default=10)
    pr.add_argument("--top_k", type=int, default=9)
    pr.add_argument("--temperature", type=float, default=0.7)
    pr.add_argument("--max_new_tokens", type=int, default=16)

    pr.add_argument("--fallback_score", type=int, default=5, help="if no valid sample at a rubric -> this score")
    pr.add_argument("--xtick_step", type=int, default=10, help="x tick interval for plots")

    pr.add_argument("--output_root", type=str, default="./consistency_results", help="base output root")
    pr.add_argument("--output_dir", type=str, default=None, help="if set, use exactly this directory (no auto timestamp)")
    pr.add_argument("--tag", type=str, default=None, help="optional tag for output folder name")

    # baselines (optional)
    pr.add_argument("--baselines_json", type=str, default=None, help="optional baselines json file")
    pr.add_argument("--baseline_top1_overall", type=float, default=None)
    pr.add_argument("--baseline_weighted_overall", type=float, default=None)
    pr.add_argument("--baseline_top1_average", type=float, default=None)
    pr.add_argument("--baseline_weighted_average", type=float, default=None)
    pr.add_argument("--dry_run", action="store_true", help="Dry run: 모델, 데이터 로딩까지만 확인하고 샘플링은 시작하지 않음")

    # ----- analyze -----
    pa = sub.add_parser("analyze", help="Analyze an existing samples json and save plots/results.")
    pa.add_argument("--bank_path", type=str, required=True, help="samples json path")
    pa.add_argument("--test_path", type=str, required=True, help="test.jsonl path")

    pa.add_argument("--fallback_score", type=int, default=5)
    pa.add_argument("--xtick_step", type=int, default=10)

    pa.add_argument("--output_root", type=str, default="./consistency_results")
    pa.add_argument("--output_dir", type=str, default=None)
    pa.add_argument("--tag", type=str, default=None)

    pa.add_argument("--baselines_json", type=str, default=None)
    pa.add_argument("--baseline_top1_overall", type=float, default=None)
    pa.add_argument("--baseline_weighted_overall", type=float, default=None)
    pa.add_argument("--baseline_top1_average", type=float, default=None)
    pa.add_argument("--baseline_weighted_average", type=float, default=None)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "run":
        run_mode(args)
    elif args.mode == "analyze":
        analyze_mode(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
