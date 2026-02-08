from transformers import Trainer
from typing import Any
import torch
import torch.nn.functional as F
from .wasserstein_number_token_loss import WassersteinNumberTokenLoss
from .number_token_loss import NumberTokenLoss

class CustomTrainer(Trainer):
    def __init__(
        self, *args,
        ntl_weight: float = 0.3,
        emo_weight: float = 0.1,
        emo_topk: int = 64,
        num_tokenizer=None,
        order_numbers=None,
        loss_type: str = "mse",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ntl_weight = float(ntl_weight)
        self.emo_weight = float(emo_weight)
        self.emo_topk = int(emo_topk)

        self.num_tokenizer = num_tokenizer
        self.order_numbers = order_numbers

        device = self.args.device
        vocab_size = self.model.config.vocab_size

        if loss_type == "was":
            self.ntl_criterion = WassersteinNumberTokenLoss(
                vocab_size=vocab_size, device=device,
                order_numbers=self.order_numbers, tokenizer=self.num_tokenizer
            )
        elif loss_type == "mse":
            self.ntl_criterion = NumberTokenLoss(
                tokenizer=self.num_tokenizer, vocab_size=vocab_size,
                device=device, loss_function=torch.nn.functional.mse_loss
            )

        self._last_logged_step = -1

    @staticmethod
    def _to_serializable(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().float().mean().item() if v.numel() > 1 else v.detach().cpu().item()
        if isinstance(v, dict):
            return {k: CustomTrainer._to_serializable(val) for k, val in v.items()}
        return v

    @staticmethod
    def _get_embedding_matrix(model) -> torch.Tensor:
        if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
            return model.get_output_embeddings().weight
        if hasattr(model, "lm_head"):
            return model.lm_head.weight
        raise ValueError("Cannot find output embedding matrix (lm_head / get_output_embeddings).")

    def _emo_loss_topk(self, logits: torch.Tensor, emo_labels: torch.Tensor, model) -> torch.Tensor:
        if getattr(model.config, "is_encoder_decoder", False):
            raise NotImplementedError("EMO loss here assumes causal LM (decoder-only).")

        # 🚫 .contiguous() 제거: 여기서 수 GB 복사가 발생할 수 있음
        logits_s = logits[:, :-1, :]                 # [B, T-1, V]
        labels_s = emo_labels[:, 1:]                 # [B, T-1]
        mask = labels_s.ne(-100)                     # [B, T-1]

        if mask.sum().item() == 0:
            return logits_s.sum() * 0.0

        E = self._get_embedding_matrix(model)        # [V, D]
        B, Tm1, V = logits_s.shape
        D = E.size(1)

        labels_safe = labels_s.masked_fill(~mask, 0) # invalid는 0으로 (원 로직 동일)
        k = min(self.emo_topk, V)

        # ✅ exact top-k probs from full softmax (원 로직 동일)
        lse = torch.logsumexp(logits_s, dim=-1)                  # [B, T-1]
        top_logits, topi = torch.topk(logits_s, k=k, dim=-1)     # [B, T-1, k]
        topv = torch.exp(top_logits - lse.unsqueeze(-1))         # [B, T-1, k]

        # exclude GT if it appears in top-k (원 로직 동일: prob=0, no renorm)
        gt = labels_safe.unsqueeze(-1)
        topv = topv.masked_fill(topi.eq(gt), 0.0)

        # apply valid-token mask (원 로직 동일)
        topv = topv * mask.unsqueeze(-1)

        # ---- 여기부터: p/q를 전체 텐서로 만들지 않고 "블록 단위"로 동일 계산 ----
        N = B * Tm1
        topi_f = topi.reshape(N, k)                                   # [N, k]
        topv_f = topv.reshape(N, k).to(dtype=E.dtype)                 # [N, k]
        labels_f = labels_safe.reshape(N)                              # [N]
        mask_f = mask.reshape(N)                                       # [N]

        # 결과: mean over valid tokens (원 로직과 동일)
        total = torch.zeros((), device=logits.device, dtype=torch.float32)
        count = mask_f.sum().to(dtype=torch.float32).clamp_min(1.0)

        block_size = 128  # 더 줄이면 peak 메모리 더 감소(속도는 느려짐)

        for s in range(0, N, block_size):
            e = min(s + block_size, N)

            idx_block = topi_f[s:e]                                    # [bs, k]
            w_block = topv_f[s:e]                                      # [bs, k]

            # q_block = sum_j w_j * normalize(E[idx_j])
            emb = E.index_select(0, idx_block.reshape(-1))             # [bs*k, D]
            emb = F.normalize(emb, p=2, dim=-1, eps=1e-12)
            emb = emb.view(e - s, k, D)                                # [bs, k, D]
            q_block = (w_block.unsqueeze(-1) * emb).sum(dim=1)         # [bs, D]
            q_block = F.normalize(q_block, p=2, dim=-1, eps=1e-12)

            # p_block = normalize(E[labels])
            p_block = E.index_select(0, labels_f[s:e])                 # [bs, D]
            p_block = F.normalize(p_block, p=2, dim=-1, eps=1e-12)

            cos = (p_block * q_block).sum(dim=-1)                      # [bs]
            emo_tok = 1.0 - cos                                         # [bs]

            m = mask_f[s:e]
            if m.any():
                total = total + emo_tok[m].to(torch.float32).sum()

        return total / count

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ntl_labels = inputs.get("ntl_labels", inputs.get("labels", None))
        emo_labels = inputs.get("emo_labels", None)

        inputs_for_super = {k: v for k, v in inputs.items() if k not in ("ntl_labels", "emo_labels")}

        # 1) 기본 CE
        base_loss, outputs = super().compute_loss(
            model, inputs_for_super, return_outputs=True, **kwargs
        )

        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("Model outputs must include 'logits'.")

        # 2) NTL
        if self.ntl_weight != 0:
            if not getattr(model.config, "is_encoder_decoder", False):
                # 🚫 .contiguous() 제거: 여기서도 수 GB 복사가 날 수 있음
                logits_ntl = logits[:, :-1, :]          # [B, T-1, V]
                labels_ntl = ntl_labels[:, 1:]          # [B, T-1]
            else:
                logits_ntl, labels_ntl = logits, ntl_labels

            ntl_loss = self.ntl_criterion(logits_ntl, labels_ntl)
        else:
            ntl_loss = logits.sum() * 0.0

        # 3) EMO
        if emo_labels is not None and self.emo_weight != 0:
            emo_loss = self._emo_loss_topk(logits, emo_labels, model)
        else:
            emo_loss = logits.sum() * 0.0

        eps = 1e-8

        # 4) 통합 로직 (원 로직 그대로)
        if self.emo_weight < 0:
            aux_loss = ntl_loss + emo_loss
            with torch.no_grad():
                aux_dynamic = base_loss.detach() / (aux_loss.detach() + eps)
            total_loss = 0.5 * (base_loss + aux_dynamic * aux_loss)

            current_mode = "dynamic_aux"
            current_aux_weight = float(aux_dynamic.item())
            current_ntl_weight = None
            current_emo_weight = None
        else:
            if self.ntl_weight < 0:
                with torch.no_grad():
                    ntl_dynamic = base_loss.detach() / (ntl_loss.detach() + eps)
                total_loss = 0.5 * (base_loss + ntl_dynamic * ntl_loss)
                current_ntl_weight = float(ntl_dynamic.item())
            else:
                total_loss = base_loss + self.ntl_weight * ntl_loss
                current_ntl_weight = float(self.ntl_weight)

            total_loss = total_loss + self.emo_weight * emo_loss
            current_mode = "static_emo"
            current_aux_weight = None
            current_emo_weight = float(self.emo_weight)

        # 5) 로깅 (원 로직 그대로)
        if self.model.training and self.state.global_step % self.args.logging_steps == 0:
            if self._last_logged_step != self.state.global_step:
                log_dict = {
                    "loss_ce": self._to_serializable(base_loss),
                    "loss_ntl": self._to_serializable(ntl_loss),
                    "loss_emo": self._to_serializable(emo_loss),
                    "loss_total": self._to_serializable(total_loss),
                    "loss_mode": current_mode,
                }
                if current_aux_weight is not None:
                    log_dict["aux_weight"] = float(current_aux_weight)
                if current_ntl_weight is not None:
                    log_dict["ntl_weight"] = float(current_ntl_weight)
                if current_emo_weight is not None:
                    log_dict["emo_weight"] = float(current_emo_weight)

                self.log(log_dict)
                self._last_logged_step = self.state.global_step

        if isinstance(outputs, dict):
            outputs["ce_loss"] = base_loss.detach()
            outputs["ntl_loss"] = ntl_loss.detach()
            outputs["emo_loss"] = emo_loss.detach()

        return (total_loss, outputs) if return_outputs else total_loss