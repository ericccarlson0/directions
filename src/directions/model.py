"""Model loading, metadata, batched scoring and residual capture.

The default backend is Hugging Face Transformers with PyTorch hooks. A second
``tiny_random`` backend builds a randomly-initialised Qwen3 of the same
architecture with a byte-level tokenizer; it needs no download, no GPU and no
network, and is what the inexpensive integration path exercises. Both backends
go through exactly the same hook, scoring and measurement code.
"""

from __future__ import annotations

import contextlib
import platform
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import Config
from .hooks import Intervention, ResidualHooks, block_output_patch, resolve_blocks
from .prompts import Prompt

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class ByteTokenizer:
    """Deterministic byte-level tokenizer for the offline test backend."""

    pad_token_id = 256
    vocab_size = 260

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids if int(i) < 256).decode("utf-8", errors="replace")

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [repr(bytes([int(i)])) if int(i) < 256 else "<pad>" for i in ids]


@dataclass
class Encoded:
    """One prompt tokenized together with its teacher-forced target."""

    input_ids: list[int]
    prompt_len: int
    target_ids: list[int]
    item_id: str

    @property
    def total_len(self) -> int:
        return len(self.input_ids)


@dataclass
class BlockPatch:
    """Additive patch applied to one block's *output* at the measured token.

    Used by the exploratory causal-validation analysis: adding ``-b_l(x)`` to
    the output of block ``l`` during a steered run removes exactly that block's
    steering-induced contribution (necessity), while adding ``+b_l(x)`` during
    an unsteered run injects it (sufficiency).
    """

    layer: int
    vectors: np.ndarray  # (n_prompts, d_model), aligned with the prompt list
    sign: float = -1.0


@dataclass
class BehaviorResult:
    """Per-example behavioural measurements for one condition."""

    item_ids: list[str] = field(default_factory=list)
    correct: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    target_logprob: np.ndarray = field(default_factory=lambda: np.zeros(0))
    target_logprob_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    logit_margin: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def summary(self) -> dict:
        return {
            "n": int(self.correct.size),
            "accuracy": float(np.mean(self.correct)) if self.correct.size else float("nan"),
            "target_logprob": float(np.mean(self.target_logprob)) if self.correct.size else float("nan"),
            "target_logprob_per_token": float(np.mean(self.target_logprob_mean))
            if self.correct.size
            else float("nan"),
            "logit_margin": float(np.mean(self.logit_margin)) if self.correct.size else float("nan"),
        }

    def metric(self, name: str) -> float:
        return {
            "accuracy": lambda: float(np.mean(self.correct)),
            "target_logprob": lambda: float(np.mean(self.target_logprob)),
            "logit_margin": lambda: float(np.mean(self.logit_margin)),
        }[name]()


class LanguageModel:
    def __init__(self, model, tokenizer, spec: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self.model.eval()
        self.blocks = resolve_blocks(model)
        self.n_layers = len(self.blocks)
        self.d_model = int(model.config.hidden_size)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(tokenizer, "eos_token_id", 0)
        self.pad_token_id = int(pad)

    # -- construction ---------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Config) -> "LanguageModel":
        return _build_tiny(cfg) if cfg.model.backend == "tiny_random" else _build_hf(cfg)

    # -- metadata -------------------------------------------------------
    def metadata(self) -> dict:
        cfg = self.model.config
        return {
            **self.spec,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_parameters": int(sum(p.numel() for p in self.model.parameters())),
            "device": str(self.device),
            "torch_dtype": str(self.dtype),
            "architecture": type(self.model).__name__,
            "vocab_size": int(getattr(cfg, "vocab_size", -1)),
            "num_attention_heads": int(getattr(cfg, "num_attention_heads", -1)),
            "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", -1)),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
        }

    def layer_index_from_fraction(self, frac: float) -> int:
        """Map a depth fraction to an intervention layer in ``[0, n_layers - 1]``."""
        return int(min(self.n_layers - 1, max(0, round(frac * self.n_layers))))

    # -- tokenization ---------------------------------------------------
    def encode(self, prompt: Prompt, max_seq_len: int) -> Encoded:
        prompt_ids = self.tokenizer.encode(prompt.text)
        target_ids = self.tokenizer.encode(prompt.target, add_special_tokens=False)
        if not target_ids:
            raise ValueError(f"empty target tokenization for {prompt.item_id!r}")
        if len(prompt_ids) + len(target_ids) > max_seq_len:
            raise ValueError(
                f"prompt {prompt.item_id!r} needs {len(prompt_ids) + len(target_ids)} tokens "
                f"> compute.max_seq_len={max_seq_len}"
            )
        return Encoded(prompt_ids + target_ids, len(prompt_ids), target_ids, prompt.item_id)

    def encode_all(self, prompts: list[Prompt], max_seq_len: int) -> list[Encoded]:
        return [self.encode(p, max_seq_len) for p in prompts]

    def _pad_batch(self, batch: list[Encoded], upto: str):
        """Right-pad a batch. ``upto`` is ``"prompt"`` or ``"total"``."""
        lens = [e.prompt_len if upto == "prompt" else e.total_len for e in batch]
        width = max(lens)
        ids = torch.full((len(batch), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, (e, n) in enumerate(zip(batch, lens)):
            ids[i, :n] = torch.tensor(e.input_ids[:n], dtype=torch.long)
            mask[i, :n] = 1
        last_prompt = torch.tensor([e.prompt_len - 1 for e in batch], dtype=torch.long)
        return ids.to(self.device), mask.to(self.device), last_prompt

    def _hook_positions(self, batch: list[Encoded]) -> torch.Tensor:
        return torch.tensor([e.prompt_len - 1 for e in batch], dtype=torch.long)

    def _intervention(self, spec, batch: list[Encoded]) -> Intervention | None:
        if spec is None:
            return None
        return Intervention(
            layer=spec.layer,
            vector=torch.as_tensor(np.asarray(spec.vector, dtype=np.float32)),
            alpha=float(spec.alpha),
            positions=self._hook_positions(batch),
        )

    # -- forward passes -------------------------------------------------
    @torch.inference_mode()
    def score(
        self,
        prompts: list[Prompt],
        intervention=None,
        batch_size: int = 8,
        max_seq_len: int = 1024,
        block_patch: "BlockPatch | None" = None,
    ) -> BehaviorResult:
        """Teacher-forced behavioural metrics under an optional intervention.

        Reported per example: exact-match of the argmax target continuation,
        summed and per-token target log-probability, and the first-target-token
        logit margin. All are exact and deterministic (no sampling).
        """
        encoded = self.encode_all(prompts, max_seq_len)
        # Sorting by length keeps padding (and the lm_head window) small.
        order = np.argsort([e.total_len for e in encoded], kind="stable")
        item_ids, correct, lp_sum, lp_mean, margins = [], [], [], [], []
        for start in range(0, len(order), batch_size):
            chunk = [encoded[int(i)] for i in order[start : start + batch_size]]
            ids, mask, _ = self._pad_batch(chunk, "total")
            width = ids.shape[1]
            keep = min(width, max(width - (e.prompt_len - 1) for e in chunk))
            iv = self._intervention(intervention, chunk)
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    ResidualHooks(self.blocks, positions=self._hook_positions(chunk),
                                  capture=False, intervention=iv)
                )
                if block_patch is not None:
                    stack.enter_context(
                        block_output_patch(
                            self.blocks[block_patch.layer],
                            self._patch_fn(block_patch, order[start : start + batch_size], chunk),
                        )
                    )
                logits = self._forward_logits(ids, mask, keep)
            offset = width - logits.shape[1]
            for row, e in enumerate(chunk):
                first = e.prompt_len - 1 - offset
                n_t = len(e.target_ids)
                sel = logits[row, first : first + n_t].to(torch.float32)
                tgt = torch.tensor(e.target_ids, device=sel.device)
                logprobs = torch.log_softmax(sel, dim=-1)
                tok_lp = logprobs.gather(-1, tgt[:, None]).squeeze(-1)
                argmax_ok = bool(torch.equal(sel.argmax(-1), tgt))
                first_logits = sel[0].clone()
                gold = float(first_logits[tgt[0]])
                first_logits[tgt[0]] = -float("inf")
                item_ids.append(e.item_id)
                correct.append(argmax_ok)
                lp_sum.append(float(tok_lp.sum()))
                lp_mean.append(float(tok_lp.mean()))
                margins.append(gold - float(first_logits.max()))
        # restore original prompt order
        inv = np.argsort(order, kind="stable")
        take = lambda xs, dt: np.asarray(xs, dtype=dt)[inv]  # noqa: E731
        return BehaviorResult(
            item_ids=[item_ids[int(i)] for i in inv],
            correct=take(correct, bool),
            target_logprob=take(lp_sum, np.float64),
            target_logprob_mean=take(lp_mean, np.float64),
            logit_margin=take(margins, np.float64),
        )

    def _patch_fn(self, patch: "BlockPatch", indices, chunk: list[Encoded]):
        vecs = torch.as_tensor(np.asarray(patch.vectors, dtype=np.float32)[np.asarray(indices)])
        positions = self._hook_positions(chunk)

        def fn(h: torch.Tensor) -> torch.Tensor:
            h = h.clone()
            rows = torch.arange(h.shape[0], device=h.device)
            h[rows, positions.to(h.device)] += (patch.sign * vecs).to(
                device=h.device, dtype=h.dtype
            )
            return h

        return fn

    def _forward_logits(self, ids, mask, keep: int) -> torch.Tensor:
        """Forward pass returning logits for the last ``keep`` positions only."""
        try:
            out = self.model(input_ids=ids, attention_mask=mask, logits_to_keep=keep)
            if out.logits.shape[1] == keep:
                return out.logits
        except TypeError:
            out = self.model(input_ids=ids, attention_mask=mask)
        return out.logits[:, -keep:]

    @torch.inference_mode()
    def capture(
        self,
        prompts: list[Prompt],
        intervention=None,
        batch_size: int = 8,
        max_seq_len: int = 1024,
        include_target: bool = False,
    ) -> np.ndarray:
        """Residual stream at the final query token for every layer.

        Returns ``(n_layers + 1, n_prompts, d_model)`` float32, in the order the
        prompts were given. Only the prompt is fed through the model unless
        ``include_target`` is set, because the measured token is the last prompt
        token in either case.
        """
        encoded = self.encode_all(prompts, max_seq_len)
        order = np.argsort([e.prompt_len for e in encoded], kind="stable")
        out = np.zeros((self.n_layers + 1, len(encoded), self.d_model), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            sel = order[start : start + batch_size]
            chunk = [encoded[int(i)] for i in sel]
            ids, mask, _ = self._pad_batch(chunk, "total" if include_target else "prompt")
            iv = self._intervention(intervention, chunk)
            with ResidualHooks(self.blocks, positions=self._hook_positions(chunk),
                               capture=True, intervention=iv) as hooks:
                self.model(input_ids=ids, attention_mask=mask)
                stacked = hooks.stacked().numpy()  # (L+1, b, d)
            out[:, sel, :] = stacked
        return out


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("run.device='cuda' but CUDA is not available")
    return torch.device(name)


def _build_hf(cfg: Config) -> LanguageModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(cfg.run.device)
    dtype = DTYPES[cfg.run.dtype]
    if device.type == "cpu" and dtype is torch.bfloat16:
        dtype = torch.float32  # bf16 on CPU is slow and poorly supported
    kw = dict(revision=cfg.model.revision, trust_remote_code=cfg.model.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        dtype=dtype,
        attn_implementation=cfg.model.attn_implementation,
        **kw,
    ).to(device)
    spec = {
        "backend": "hf",
        "name_or_path": cfg.model.name_or_path,
        "revision": cfg.model.revision,
        "requested_dtype": cfg.run.dtype,
        "attn_implementation": cfg.model.attn_implementation,
    }
    return LanguageModel(model, tokenizer, spec)


def _build_tiny(cfg: Config) -> LanguageModel:
    """Randomly-initialised Qwen3 with a byte tokenizer: no download, no GPU."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    t = cfg.model.tiny
    tokenizer = ByteTokenizer()
    conf = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=t.hidden_size,
        intermediate_size=t.intermediate_size,
        num_hidden_layers=t.num_hidden_layers,
        num_attention_heads=t.num_attention_heads,
        num_key_value_heads=t.num_key_value_heads,
        head_dim=t.head_dim,
        max_position_embeddings=t.max_position_embeddings,
        attn_implementation="eager",
    )
    torch.manual_seed(t.init_seed)
    model = Qwen3ForCausalLM(conf)
    device = _resolve_device(cfg.run.device)
    model = model.to(device=device, dtype=torch.float32)
    spec = {
        "backend": "tiny_random",
        "name_or_path": f"tiny_random_qwen3(seed={t.init_seed})",
        "revision": None,
        "requested_dtype": "float32",
        "attn_implementation": "eager",
    }
    return LanguageModel(model, tokenizer, spec)
