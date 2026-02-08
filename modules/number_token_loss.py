import torch
import torch.nn.functional as F
from torch._tensor import Tensor
from .number_tokenizer import NumberEncodingTokenizer

class NumberTokenSelector:
    def __init__(self, tokenizer: NumberEncodingTokenizer, vocab_size, device):
        self.tokenizer = tokenizer
        self.nvocab = torch.full((vocab_size,), float("nan"), device=device)

        hashed_num_tokens = set(self.tokenizer.get_num_tokens())
        for token, tid in self.tokenizer.get_vocab().items():
            if token in hashed_num_tokens:
                try:
                    val = self.tokenizer.decode_number_token(token, ignore_order=True)
                    # 수치형 안전장치: NaN/Inf 제외
                    tval = torch.tensor(val, device=device, dtype=torch.float32)
                    if not torch.isfinite(tval):
                        continue
                    self.nvocab[tid] = float(val)
                except Exception:
                    pass

        # (1) 숫자 토큰: 유한값만 허용
        self.number_token_mask = torch.isfinite(self.nvocab)  # [V]
        self.number_token_indices = torch.nonzero(self.number_token_mask, as_tuple=False).squeeze(-1)  # [N_num]
        # (2) 전체 숫자값 테이블 (선택축으로 축소된 것)
        self.values_all = torch.nan_to_num(
            self.nvocab[self.number_token_indices], nan=0.0, posinf=0.0, neginf=0.0
        )  # [N_num]

        # (3) digit(1~9) 전용 마스크/값 (선택축 기준)
        vals = self.values_all
        is_int = (vals == torch.round(vals))
        # 라벨이 1~9 라고 하셨으므로 그 구간만 선택
        self.mask_digits = (is_int & (vals >= 1) & (vals <= 9))  # [N_num] bool
        self.values_digits = vals[self.mask_digits]              # [N_digit]

    def select_number_tokens(self, logits: Tensor):
        """
        logits: [B, T, V]
        returns:
          logits_num: [B, T, N_num]
          number_token_mask: [V] bool
        """
        logits = logits[:, :, self.number_token_mask]
        return logits, self.number_token_mask


class NumberTokenLoss:
    def __init__(self, tokenizer: NumberEncodingTokenizer, vocab_size: int, device,
                 loss_function=F.mse_loss, weight=0.5):
        self.loss_function = loss_function
        self.weight = weight
        self.selector = NumberTokenSelector(tokenizer, vocab_size, device)
        self.nvocab = self.selector.nvocab  # [V]

    def forward(self, logits: Tensor, labels: Tensor):
        if logits.numel() == 0:
            raise ValueError("Logits passed to the NumberTokenLoss are empty!")
        if labels.numel() == 0:
            raise ValueError("Labels passed to the NumberTokenLoss are empty!")

        # 1) 숫자 토큰 로짓만 선택
        logits_num, number_tokens_mask = self.selector.select_number_tokens(logits)  # [B,T,N_num], [V]
        # 2) 소프트맥스 (안정화)
        softmaxed = F.softmax(torch.clamp(logits_num, min=-50, max=50), dim=-1)      # [B,T,N_num]

        # 3) 전체 기대값 (폴백용)
        values_all = self.selector.values_all                                        # [N_num]
        if values_all.numel() != logits_num.size(-1):
            raise RuntimeError(
                f"values_all length ({values_all.numel()}) != logits_num last dim ({logits_num.size(-1)})"
            )
        yhat_all = torch.sum(softmaxed * values_all, dim=-1)                         # [B,T]
        yhat_all = torch.nan_to_num(yhat_all, nan=0.0, posinf=0.0, neginf=0.0)

        # 4) digit 전용 기대값 (부분분포 재정규화)
        mask_digits = self.selector.mask_digits                                      # [N_num] bool
        values_digits = self.selector.values_digits                                  # [N_digit]
        if values_digits.numel() == 0:
            # 숫자 토큰 중 1~9가 전혀 없다면, 전체 기대값으로만 계산
            yhat_digits = yhat_all
            p_sum = torch.zeros_like(yhat_all)
        else:
            p_digits = softmaxed[..., mask_digits]                                   # [B,T,N_digit]
            p_sum = p_digits.sum(-1, keepdim=True)                                   # [B,T,1]
            # p_sum==0인 경우 재정규화가 무의미하므로 clamp
            p_digits = p_digits / p_sum.clamp_min(1e-12)
            yhat_digits = torch.sum(p_digits * values_digits, dim=-1)                # [B,T]
            yhat_digits = torch.nan_to_num(yhat_digits, nan=0.0, posinf=0.0, neginf=0.0)

        # 5) 라벨의 실제 숫자값 (nvocab에서)
        safe_labels = labels.masked_fill(labels == -100, 0)                          # 인덱싱 안전용
        y_all = self.nvocab[safe_labels]                                             # [B,T]
        y_all = torch.nan_to_num(y_all, nan=0.0, posinf=0.0, neginf=0.0)

        # 6) 유효 라벨 마스크: (-100 아님) AND (라벨이 1~9)
        is_digit_label = (y_all == torch.round(y_all)) & (y_all >= 1) & (y_all <= 9) # [B,T]
        valid_label_mask = (labels != -100) & is_digit_label

        if valid_label_mask.sum() == 0:
            return (logits_num * 0.0).sum()

        # 7) 최종 yhat: digit 라벨에는 digit 기대값, 그 외(안 나와야 함)에는 전체 기대값 폴백
        #    또한 p_sum==0 (digits에 질량이 전혀 없는) 위치는 전체 기대값으로 폴백
        use_digits = is_digit_label & (p_sum.squeeze(-1) > 1e-8)                     # [B,T]
        yhat = torch.where(use_digits, yhat_digits, yhat_all)                        # [B,T]

        # 8) 손실 계산 (유효 위치만)
        loss = self.loss_function(yhat[valid_label_mask], y_all[valid_label_mask])

        # 9) 최종 안전장치
        if torch.isnan(loss) or torch.isinf(loss):
            return (logits_num * 0.0).sum()

        return loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
