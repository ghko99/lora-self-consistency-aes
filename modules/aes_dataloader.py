from transformers import DataCollatorForLanguageModeling
import unicodedata
from typing import List, Dict, Union
import torch

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# =========================
# Collator (동적 패딩 + 질문 마스킹)
# =========================
class LlamaAESCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, max_seq_length=2048):
        super().__init__(tokenizer, mlm=False)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eot_token = tokenizer.eos_token
        self.max_seq_length = max_seq_length

    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]):
        merged_sequences = [
            f"{normalize_text(ex['instruction'])}{normalize_text(ex['output'])}{self.eot_token}"
            for ex in examples
        ]
        batch = self.tokenizer(
            merged_sequences,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        labels = batch["input_ids"].clone()

        for i, ex in enumerate(examples):
            q_ids = self.tokenizer(
                normalize_text(ex["instruction"]),
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )["input_ids"][0]
            q_len = q_ids.size(0)
            labels[i, :q_len] = -100

        labels[labels == self.pad_token_id] = -100
        return {"input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": labels}

class LlamaAESCollatorMTL(DataCollatorForLanguageModeling):
    """
    - labels: CE 학습용 (instruction은 -100 마스킹, output 전체 학습)
    - ntl_labels: NTL 학습용 (output 중 '점수 파트(처음 17 토큰)'만 남기고 나머지는 -100)
    """
    def __init__(self, tokenizer, max_seq_length: int = 2048, score_token_len: int = 16):
        super().__init__(tokenizer, mlm=False)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eot_token = tokenizer.eos_token
        self.max_seq_length = int(max_seq_length)
        self.score_token_len = int(score_token_len)

    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]):
        # 1) instruction + output + eos
        merged_sequences = [
            f"{normalize_text(ex['instruction'])}{normalize_text(ex['output'])}{self.eot_token}"
            for ex in examples
        ]

        batch = self.tokenizer(
            merged_sequences,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # 2) CE용 labels 만들기
        labels = batch["input_ids"].clone()

        # instruction 길이 계산해서 instruction 구간 마스킹
        q_lens = []
        for i, ex in enumerate(examples):
            q_ids = self.tokenizer(
                normalize_text(ex["instruction"]),
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )["input_ids"][0]
            q_len = int(q_ids.size(0))
            q_lens.append(q_len)

            labels[i, :q_len] = -100  # instruction은 loss에서 제외

        # pad는 loss 제외
        labels[batch["attention_mask"] == 0] = -100


        # 3) NTL용 ntl_labels 만들기: output의 점수 파트(처음 17 토큰)만 남김
        ntl_labels = torch.full_like(labels, -100)
        emo_labels = torch.full_like(labels, -100)
        
        B, T = labels.shape

        for i in range(B):
            start = q_lens[i]  # output 시작 위치 (= instruction 끝)
            end = min(start + self.score_token_len, T)
            if start < T and start < end:
                ntl_labels[i, start:end] = labels[i, start:end]
            if end < T:
                emo_labels[i, end:] = labels[i, end:]
            

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,           # CE 전체용
            "ntl_labels": ntl_labels,   # NTL 점수용
            "emo_labels": emo_labels,   # EMO Feedback용 
        }
