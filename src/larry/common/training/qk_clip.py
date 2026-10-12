import torch
from torch import nn

from larry.common.metrics.metrics import MetricDict, ScalarMetricEntry
from larry.common.training import trainers, callbacks


class QKModule:
    q_proj: nn.Linear
    k_proj: nn.Linear
    d_queries: int
    qk_probe_active: bool = False
    qk_max_logit: torch.Tensor | None = None


class QKClipHandler:
    def __init__(self, model: nn.Module, tau: float, probe_every: int, alpha: float = 0.5) -> None:
        if tau is None or tau <= 0:
            raise ValueError(f"qk_clip tau must be positive, got {tau}")

        self.tau = tau
        self.probe_every = max(probe_every, 1)
        self.alpha = alpha

        self.modules: list[tuple[str, QKModule]] = []

        for name, m in model.named_modules():
            if isinstance(m, QKModule):
                if not m.q_proj.weight.requires_grad:
                    # don't clip frozen modules
                    continue
                self.modules.append((name, m))

        self.armed = False

    def __len__(self) -> int:
        return len(self.modules)

    def maybe_arm(self, step: int) -> None:
        """Turn the probe on for this step's forward pass, if it is a sampled step."""
        want = (step % self.probe_every) == 0
        if want == self.armed:
            return

        for _, m in self.modules:
            m.qk_probe_active = want
            if want:
                m.qk_max_logit = None

        self.armed = want

    def disarm(self) -> None:
        """Turn the probe off. Called at the end of every apply()."""
        if not self.armed:
            return

        for _, m in self.modules:
            m.qk_probe_active = False

        self.armed = False

    @torch.no_grad()
    def apply(self) -> MetricDict | None:
        """Rescale w_q / w_k for every head whose observed max logit exceeds tau."""
        if not self.armed:
            return None

        n_clipped = 0
        n_heads_total = 0
        max_seen = float("-inf")
        min_eta = 1.0

        for _, m in self.modules:
            if m.qk_max_logit is None:
                continue

            qk = m.qk_max_logit.float()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(qk, op=torch.distributed.ReduceOp.MAX)
            n_heads_total += qk.numel()

            max_seen = max(max_seen, float(qk.max()))

            w_q = m.q_proj.weight  # (n_heads*dq, d_model)
            w_k = m.k_proj.weight  # (n_groups*dq, d_model)

            eta = (self.tau / qk.clamp_min(1e-12)).clamp(max=1.0)  # (n_heads,)
            if bool((eta < 1.0).any()):
                n_queries = w_q.shape[0] // m.d_queries
                n_keys = max(w_k.shape[0] // m.d_queries, 1)

                n_rep = max(n_queries // n_keys, 1)

                # GQA: a k block serves n_rep q heads -> take the strongest clip in the group.
                g_grouped = eta[:n_queries].view(n_keys, n_rep).min(dim=1).values
                gk = g_grouped ** (1.0 - self.alpha)  # (n_keys,)
                gq = eta[:n_queries] / gk.repeat_interleave(n_rep)  # (n_queries,); gq * gk == eta exactly

                w_q.mul_(gq.repeat_interleave(m.d_queries).unsqueeze(1).to(w_q.dtype))
                w_k.mul_(gk.repeat_interleave(m.d_queries).unsqueeze(1).to(w_k.dtype))

                if m.q_proj.bias is not None:
                    m.q_proj.bias.mul_(gq.repeat_interleave(m.d_queries).to(w_q.dtype))
                if m.k_proj.bias is not None:
                    m.k_proj.bias.mul_(gk.repeat_interleave(m.d_queries).to(w_k.dtype))

                n_clipped += int((eta < 1.0).sum())
                min_eta = min(min_eta, float(eta.min()))

        self.disarm()

        step_metrics = MetricDict()
        step_metrics.add_metrics([
            ScalarMetricEntry(tag="qk_clip/max_logit", scalar=max_seen if max_seen > float("-inf") else float("nan")),
            ScalarMetricEntry(tag="qk_clip/heads_clipped", scalar=float(n_clipped)),
            ScalarMetricEntry(tag="qk_clip/heads_total", scalar=float(n_heads_total)),
            ScalarMetricEntry(tag="qk_clip/min_eta", scalar=float(min_eta)),
        ])
        return step_metrics


class QKClipCallback(callbacks.TrainingCallback):
    """Arms the probe before the forward, applies the clip after optimizer.step()."""

    def __init__(self, handler: QKClipHandler):
        self.handler = handler

    def on_step_begin(self, trainer: trainers.TrainerBase, step: int) -> MetricDict | None:
        self.handler.maybe_arm(step)

    def on_step_end(self, trainer: trainers.TrainerBase, train_loss: torch.Tensor, step: int) -> MetricDict | None:
        return self.handler.apply()
