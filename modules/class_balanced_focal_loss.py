import torch
import torch.nn.functional as F
from .number_tokenizer import NumberEncodingTokenizer


class ClassBalancedFocalLoss:
    """Class-Balanced Focal Loss for score tokens (1-9).

    Applies focal loss only at valid score token positions (ntl_labels != -100),
    restricted to digit token logits (1-9). Class weights are computed via the
    effective number of samples approach from Cui et al. (2019).
    """

    def __init__(
        self,
        tokenizer: NumberEncodingTokenizer,
        vocab_size: int,
        class_counts: torch.Tensor,
        device,
        beta: float = 0.9999,
        gamma: float = 2.0,
    ):
        self.gamma = gamma

        # Build digit token ID mapping: score value (1-9) -> list of vocab IDs
        self.digit_token_ids = {}  # {int_score: [token_ids]}
        nvocab = torch.full((vocab_size,), float("nan"), device="cpu")

        hashed_num_tokens = set(tokenizer.get_num_tokens())
        for token, tid in tokenizer.get_vocab().items():
            if token in hashed_num_tokens:
                try:
                    val = tokenizer.decode_number_token(token, ignore_order=True)
                    tval = torch.tensor(val, dtype=torch.float32)
                    if not torch.isfinite(tval):
                        continue
                    nvocab[tid] = float(val)
                except Exception:
                    pass

        # Collect token IDs for each digit score 1-9
        for tid in range(vocab_size):
            v = nvocab[tid].item()
            if v != v:  # NaN check
                continue
            iv = int(round(v))
            if 1 <= iv <= 9 and abs(v - iv) < 1e-6:
                self.digit_token_ids.setdefault(iv, []).append(tid)

        self.num_classes = 9

        # Flatten to a single tensor of digit token IDs for fast indexing
        flat_ids = []
        id_to_score_idx = {}
        for score_val in range(1, 10):
            for tid in self.digit_token_ids.get(score_val, []):
                id_to_score_idx[tid] = score_val - 1
                flat_ids.append(tid)

        self.digit_token_id_tensor = torch.tensor(
            sorted(set(flat_ids)), dtype=torch.long, device=device
        )
        # Map from token_id to score index (0-8)
        self._tid_to_score_idx = torch.full(
            (vocab_size,), -1, dtype=torch.long, device=device
        )
        for tid, sidx in id_to_score_idx.items():
            self._tid_to_score_idx[tid] = sidx

        # Compute class-balanced weights using effective number
        # class_counts shape: [9], counts for scores 1-9
        class_counts = class_counts.float().to(device)
        effective_num = (1.0 - beta ** class_counts) / (1.0 - beta)
        weights = 1.0 / effective_num.clamp_min(1e-8)
        # Normalize so weights sum to num_classes
        weights = weights / weights.sum() * self.num_classes
        self.class_weights = weights  # [9]

    def extract_digit_logits(
        self, logits: torch.Tensor, labels: torch.Tensor
    ):
        """Step 1: Extract digit logits at valid score-token positions.

        Gradient flows through this step (logits → digit_logits).
        Kept outside checkpoint so the checkpoint only saves the tiny
        [M, N_digit] tensor instead of the full [B, T, V] logits.

        Args:
            logits: [B, T, V]
            labels: [B, T] with -100 at non-score positions

        Returns:
            (digit_logits [M, N_digit], score_indices [M])
            or (None, None) if no valid score positions exist.
        """
        valid_mask = labels.ne(-100)  # [B, T]
        if valid_mask.sum() == 0:
            return None, None

        valid_labels = labels[valid_mask]                      # [N_valid]
        score_indices = self._tid_to_score_idx[valid_labels]  # [N_valid]

        digit_mask = score_indices.ne(-1)
        if digit_mask.sum() == 0:
            return None, None

        score_indices = score_indices[digit_mask]              # [M]

        # [B, T, N_digit] → [M, N_digit]  (tiny tensors, gradient kept)
        # Use gather(..., sparse_grad=True) so backward can propagate sparse
        # gradients for the CBFL branch instead of allocating a dense [B, T, V]
        # temporary buffer solely for token-id indexing.
        gather_index = self.digit_token_id_tensor.view(1, 1, -1).expand(
            logits.size(0), logits.size(1), -1
        )
        digit_logits = torch.gather(
            logits, dim=2, index=gather_index, sparse_grad=True
        )  # [B, T, N_digit]
        digit_logits = digit_logits[valid_mask][digit_mask]      # [M, N_digit]

        return digit_logits, score_indices

    def compute_focal_loss(
        self, digit_logits: torch.Tensor, score_indices: torch.Tensor
    ) -> torch.Tensor:
        """Step 2: Compute focal loss from pre-extracted digit logits.

        Only tiny [M, N_digit] tensors flow through this step.
        Suitable for gradient checkpointing with negligible recompute cost.

        Args:
            digit_logits: [M, N_digit] — digit-token logits at score positions
            score_indices: [M]         — target score class index (0-8)

        Returns:
            Scalar focal loss.
        """
        # Softmax over digit tokens
        probs = F.softmax(digit_logits, dim=-1)  # [M, N_digit_tokens]

        # Build score-level probs: [M, 9]
        score_level_probs = torch.zeros(
            digit_logits.size(0), self.num_classes,
            device=digit_logits.device, dtype=digit_logits.dtype,
        )

        digit_score_map = self._tid_to_score_idx[self.digit_token_id_tensor]  # [N_digit_tokens]
        for s in range(self.num_classes):
            s_mask = digit_score_map.eq(s)
            if s_mask.any():
                score_level_probs[:, s] = probs[:, s_mask].sum(dim=-1)

        # p_t = probability of the target score class
        p_t = score_level_probs.gather(1, score_indices.unsqueeze(1)).squeeze(1)  # [M]
        p_t = p_t.clamp(1e-8, 1.0)

        focal_weight = (1.0 - p_t) ** self.gamma   # [M]
        alpha_t = self.class_weights[score_indices]  # [M]
        loss = -alpha_t * focal_weight * torch.log(p_t)  # [M]
        return loss.mean()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced focal loss on score token positions.

        Args:
            logits: shifted logits [B, T, V] (already shifted by caller)
            labels: shifted ntl_labels [B, T] (-100 for non-score positions)

        Returns:
            Scalar loss tensor.
        """
        digit_logits, score_indices = self.extract_digit_logits(logits, labels)
        if digit_logits is None:
            return logits.sum() * 0.0
        return self.compute_focal_loss(digit_logits, score_indices)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
