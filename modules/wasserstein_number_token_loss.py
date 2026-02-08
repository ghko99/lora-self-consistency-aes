import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import logging

class WassersteinNumberTokenLoss:
    def __init__(self, tokenizer, vocab_size: int, device, order_numbers: bool):
        self.tokenizer = tokenizer

        hashed_num_tokens = set(self.tokenizer.get_num_tokens())

        # nvocab[vocab_id] = numeric value if number-token else nan
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        for token, vid in self.tokenizer.get_vocab().items():
            if token in hashed_num_tokens:
                self.nvocab[vid] = self.tokenizer.decode_number_token(token, ignore_order=False)

        number_token_mask = ~torch.isnan(self.nvocab)
        self.number_token_indices = torch.nonzero(number_token_mask, as_tuple=False).squeeze(-1)

        if order_numbers:
            logging.info("Sorting number tokens by numerical value...")
            vals = self.nvocab[self.number_token_indices]
            sorted_vals, sorted_idx = torch.sort(vals)
            self.number_token_indices = self.number_token_indices[sorted_idx]
            self.number_token_values = sorted_vals
        else:
            self.number_token_values = self.nvocab[self.number_token_indices]

        # 🔑 vocab_id -> position in [0..N-1], non-number = -1
        N = self.number_token_indices.numel()
        self.id_to_numpos = torch.full((vocab_size,), -1, device=device, dtype=torch.long)
        self.id_to_numpos[self.number_token_indices] = torch.arange(N, device=device, dtype=torch.long)

    def forward(
        self,
        logits: Tensor,                 # (B, S, V)
        labels: Tensor,                 # (B, S)
        smoothed_labels: Optional[Tensor] = None,
        chunk_size: Optional[int] = None,
        softmax_dtype: Optional[torch.dtype] = None,  # e.g. torch.float32 for stability
    ) -> Tensor:
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        B, S, V = logits.shape
        N = self.number_token_indices.numel()

        ignore_mask = (labels == -100)
        labels_safe = labels.clone()
        labels_safe[ignore_mask] = 0  # safe indexing

        # numpos: (B,S), 숫자 토큰이면 [0..N-1], 아니면 -1
        numpos = self.id_to_numpos[labels_safe]
        valid = (~ignore_mask) & (numpos >= 0)

        if not torch.any(valid):
            # 그래프를 유지하면서 0 반환
            return logits.sum() * 0.0

        # 숫자 토큰 logits만 추출: (B,S,N)
        logits_num = logits.index_select(dim=-1, index=self.number_token_indices)

        # valid 위치만 모으기: (M,N)
        logits_num = logits_num[valid]
        t = numpos[valid]  # (M,)

        # (옵션) 안정성 위해 softmax 계산 dtype 지정
        if softmax_dtype is None:
            softmax_dtype = logits_num.dtype

        def compute_chunk(logits_chunk: Tensor, t_chunk: Tensor, label_chunk: Optional[Tensor]):
            # log_softmax -> exp로 probabilities, (M,N)
            logp = F.log_softmax(logits_chunk.to(softmax_dtype), dim=-1)
            p = logp.exp()  # (M,N)

            cdf_x = torch.cumsum(p, dim=-1)  # (M,N)

            if label_chunk is None:
                # hard label: target CDF는 "계단(step)" -> cdf_y를 만들지 않고 공식으로 계산
                # dist = sum_k |cdf_x[k] - step_t[k]|
                # step_t[k] = 0 (k<t), 1 (k>=t)
                # dist = sum_{k<t} cdf_x[k] + sum_{k>=t} (1 - cdf_x[k])
                #      = total + (N - t) - 2*suffix_sum_from_t(cdf_x)
                total = cdf_x.sum(dim=-1)  # (M,)

                # area[k] = sum_{j<=k} cdf_x[j]
                area = torch.cumsum(cdf_x, dim=-1)  # (M,N)

                t_minus1 = (t_chunk - 1).clamp(min=0)
                area_before = torch.where(
                    t_chunk == 0,
                    torch.zeros_like(total),
                    area.gather(1, t_minus1.unsqueeze(1)).squeeze(1)
                )
                suffix = total - area_before
                dist = total + (N - t_chunk).to(total.dtype) - 2.0 * suffix
                return dist

            else:
                # smoothed label: label_chunk shape (M,N), 이미 숫자 토큰 공간이라고 가정
                cdf_y = torch.cumsum(label_chunk.to(cdf_x.dtype), dim=-1)
                dist = torch.sum(torch.abs(cdf_x - cdf_y), dim=-1)
                return dist

        # smoothed_labels 처리: 전체 vocab 분포면 숫자부분만 뽑아서 valid만 모음
        label_valid = None
        if smoothed_labels is not None:
            if smoothed_labels.shape[-1] == V:
                sl_num = smoothed_labels.index_select(dim=-1, index=self.number_token_indices)  # (B,S,N)
                label_valid = sl_num[valid]  # (M,N)
            elif smoothed_labels.shape[-1] == N:
                label_valid = smoothed_labels[valid]  # (M,N)
            else:
                raise ValueError(
                    f"smoothed_labels last dim must be vocab_size({V}) or num_number_tokens({N}), "
                    f"but got {smoothed_labels.shape[-1]}"
                )

        # chunking으로 peak 메모리 더 절감 가능
        if chunk_size is None:
            dist = compute_chunk(logits_num, t, label_valid)
            return dist.mean()

        # chunk loop
        total_sum = logits_num.new_zeros(())
        total_cnt = 0

        M = logits_num.size(0)
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            logits_c = logits_num[start:end]
            t_c = t[start:end]
            lab_c = None if label_valid is None else label_valid[start:end]

            dist_c = compute_chunk(logits_c, t_c, lab_c)
            total_sum = total_sum + dist_c.sum()
            total_cnt += dist_c.numel()

        return total_sum / total_cnt

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
