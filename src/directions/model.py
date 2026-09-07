"""Model backend: Hugging Face Transformers + PyTorch hooks.

Residual-stream indexing (see docs/EXPERIMENT.md): for ``L`` blocks there are
``L+1`` read points. ``resid[l]`` for ``0 <= l <= L-1`` is the *input* of block
``l`` (captured/modified in a forward pre-hook); ``resid[L]`` is the output of
the last block before the final norm (captured/modified in a forward hook).

Every forward pass is teacher-forced over ``prompt + target``. Behavioural
metrics are computed from the logits at the target positions only; residuals
are gathered at the final query token *inside the hooks*, so full activation
tensors are never retained.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .config import ModelConfig
from .prompts import Prompt
from .seeds import derive_seed

# --------------------------------------------------------------------------- #
# Tokenisers
# --------------------------------------------------------------------------- #


class CharTokenizer:
    """Character-level tokeniser for the toy backend (no downloads needed)."""

    def __init__(self) -> None:
        chars = [chr(c) for c in range(32, 127)] + list("àâäçéèêëîïôöùûüÿœæÀÂÉÈÊËÎÏÔÙÛÜŒÆ")
        self.pad_token_id = 0
        self.unk_token_id = 1
        self._itos = ["<pad>", "<unk>"] + chars
        self._stoi = {c: i for i, c in enumerate(self._itos)}
        self.name = "char"

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._stoi.get(c, self.unk_token_id) for c in text]


class HFTokenizer:
    def __init__(self, name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(name)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self.pad_token_id = int(self._tok.pad_token_id)
        self.name = type(self._tok).__name__

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=add_special_tokens))


# --------------------------------------------------------------------------- #
# Interventions and results
# --------------------------------------------------------------------------- #


@dataclass
class Intervention:
    """Add ``scale * vectors`` to ``resid[layer]`` at each example's query token.

    ``vectors`` is either a single direction of shape ``(d,)`` shared by all
    examples or one vector per example, shape ``(N, d)`` (indexed like the
    prompt list passed to :meth:`ModelBackend.run`).
    """

    layer: int
    vectors: np.ndarray
    scale: float = 1.0


@dataclass
class ForwardResult:
    logprob_sum: np.ndarray  # (N,) teacher-forced log-probability of the target
    logprob_per_token: np.ndarray  # (N,) logprob_sum / n_target_tokens
    exact_match: np.ndarray  # (N,) bool: argmax equals target at every target position
    first_token_margin: np.ndarray  # (N,) gold logit minus best competing logit, first target token
    n_target_tokens: np.ndarray  # (N,)
    residuals: np.ndarray | None  # (L+1, N, d) float32 at the final query token, if captured

    def metrics_dict(self) -> dict[str, float]:
        return {
            "logprob_per_token_mean": float(np.mean(self.logprob_per_token)),
            "logprob_sum_mean": float(np.mean(self.logprob_sum)),
            "accuracy": float(np.mean(self.exact_match)),
            "first_token_margin_mean": float(np.mean(self.first_token_margin)),
            "n": int(self.logprob_sum.shape[0]),
        }


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype:
    table = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype not in table:
        raise ValueError(f"unsupported dtype {dtype!r}")
    return table[dtype]


class ModelBackend:
    """A causal LM with residual capture and intervention hooks."""

    def __init__(self, cfg: ModelConfig, run_seed: int = 0) -> None:
        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        self.dtype = _resolve_dtype(cfg.dtype)
        if cfg.backend == "hf":
            self.model, self.tokenizer = self._load_hf(cfg.name)
        elif cfg.backend == "toy":
            self.model, self.tokenizer = self._build_toy(cfg, run_seed)
        else:
            raise ValueError(f"unknown backend {cfg.backend!r}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.blocks = self._find_blocks(self.model)
        self.n_layers = len(self.blocks)
        self.hidden_size = int(self.model.config.hidden_size)
        self._session: _HookSession | None = None
        self._register_hooks()

    # ---- loading ---------------------------------------------------------- #

    def _load_hf(self, name: str):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(name, dtype=self.dtype)
        model.to(self.device)
        return model, HFTokenizer(name)

    def _build_toy(self, cfg: ModelConfig, run_seed: int):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        tok = CharTokenizer()
        t = cfg.toy
        conf = Qwen3Config(
            vocab_size=tok.vocab_size,
            hidden_size=t.hidden_size,
            intermediate_size=t.intermediate_size,
            num_hidden_layers=t.num_layers,
            num_attention_heads=t.num_heads,
            num_key_value_heads=t.num_kv_heads,
            head_dim=t.hidden_size // t.num_heads,
            max_position_embeddings=4096,
            tie_word_embeddings=True,
            pad_token_id=tok.pad_token_id,
        )
        torch.manual_seed(derive_seed(run_seed, "toy_model_init"))
        model = Qwen3ForCausalLM(conf).to(self.dtype).to(self.device)
        return model, tok

    @staticmethod
    def _find_blocks(model: torch.nn.Module) -> torch.nn.ModuleList:
        for path in ("model.layers", "transformer.h", "model.decoder.layers"):
            obj: Any = model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
            except AttributeError:
                continue
            if isinstance(obj, torch.nn.ModuleList):
                return obj
        raise ValueError("could not locate the transformer block list on this model")

    def metadata(self) -> dict[str, Any]:
        n_params = int(sum(p.numel() for p in self.model.parameters()))
        meta = {
            "backend": self.cfg.backend,
            "name": self.cfg.name if self.cfg.backend == "hf" else f"toy:{self.cfg.toy}",
            "architecture": type(self.model).__name__,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": int(self.model.config.vocab_size),
            "n_parameters": n_params,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "tokenizer": self.tokenizer.name,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        rev = getattr(self.model.config, "_commit_hash", None)
        if rev:
            meta["hf_commit_hash"] = rev
        if self.device.type == "cuda":
            meta["gpu"] = torch.cuda.get_device_name(self.device)
        meta["platform"] = platform.platform()
        return meta

    # ---- hooks ------------------------------------------------------------ #

    def _register_hooks(self) -> None:
        L = self.n_layers
        for l, block in enumerate(self.blocks):
            block.register_forward_pre_hook(self._make_pre_hook(l), with_kwargs=True)
        self.blocks[L - 1].register_forward_hook(self._make_post_hook(L))

    def _make_pre_hook(self, layer: int):
        def hook(module, args, kwargs):
            if self._session is None:
                return None
            if args:
                hidden = args[0]
                new = self._session.process(layer, hidden)
                if new is hidden:
                    return None
                return (new,) + tuple(args[1:]), kwargs
            hidden = kwargs["hidden_states"]
            new = self._session.process(layer, hidden)
            if new is hidden:
                return None
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = new
            return args, kwargs

        return hook

    def _make_post_hook(self, layer: int):
        def hook(module, args, output):
            if self._session is None:
                return None
            if isinstance(output, tuple):
                hidden = output[0]
                new = self._session.process(layer, hidden)
                if new is hidden:
                    return None
                return (new,) + tuple(output[1:])
            new = self._session.process(layer, output)
            return None if new is output else new

        return hook

    # ---- tokenisation ----------------------------------------------------- #

    def target_token_count(self, target_text: str) -> int:
        return len(self.tokenizer.encode(target_text, add_special_tokens=False))

    def _encode_prompt(self, p: Prompt) -> tuple[list[int], list[int]]:
        prompt_ids = self.tokenizer.encode(p.prompt, add_special_tokens=True)
        target_ids = self.tokenizer.encode(p.target, add_special_tokens=False)
        if not target_ids:
            raise ValueError(f"empty target for prompt {p.query}")
        return prompt_ids, target_ids

    # ---- forward ---------------------------------------------------------- #

    @torch.inference_mode()
    def run(
        self,
        prompts: Sequence[Prompt],
        interventions: Sequence[Intervention] = (),
        capture: bool = False,
        batch_size: int | None = None,
    ) -> ForwardResult:
        """Teacher-forced forward over ``prompts`` with optional interventions/capture."""
        bs = batch_size or self.cfg.batch_size
        outs: list[ForwardResult] = []
        for start in range(0, len(prompts), bs):
            batch = prompts[start : start + bs]
            sliced = [
                Intervention(iv.layer, iv.vectors if iv.vectors.ndim == 1 else iv.vectors[start : start + bs], iv.scale)
                for iv in interventions
            ]
            outs.append(self._run_batch(batch, sliced, capture))
        return _concat_results(outs)

    def _run_batch(self, prompts: Sequence[Prompt], interventions: Sequence[Intervention], capture: bool) -> ForwardResult:
        encoded = [self._encode_prompt(p) for p in prompts]
        B = len(encoded)
        lengths = [len(a) + len(b) for a, b in encoded]
        T = max(lengths)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((B, T), pad, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        query_pos = torch.zeros(B, dtype=torch.long)
        n_tgt = torch.tensor([len(b) for _, b in encoded], dtype=torch.long)
        Tt = int(n_tgt.max())
        target_ids = torch.full((B, Tt), pad, dtype=torch.long)
        for i, (a, b) in enumerate(encoded):
            ids = a + b
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attn[i, : len(ids)] = 1
            query_pos[i] = len(a) - 1
            target_ids[i, : len(b)] = torch.tensor(b)
        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)
        query_pos_dev = query_pos.to(self.device)

        session = _HookSession(
            n_layers=self.n_layers,
            query_pos=query_pos_dev,
            interventions=interventions,
            capture=capture,
            device=self.device,
        )
        self._session = session
        try:
            base_out = self.model.model(input_ids=input_ids, attention_mask=attn)
        finally:
            self._session = None
        hidden = base_out.last_hidden_state  # (B, T, d) after final norm

        # positions whose logits predict target token j: query_pos + j
        pos = (query_pos_dev[:, None] + torch.arange(Tt, device=self.device)[None, :]).clamp(max=T - 1)
        gathered = torch.gather(hidden, 1, pos[:, :, None].expand(-1, -1, hidden.shape[-1]))
        logits = self.model.lm_head(gathered).float()  # (B, Tt, V)
        logprobs = torch.log_softmax(logits, dim=-1)
        tgt_dev = target_ids.to(self.device)
        valid = (torch.arange(Tt, device=self.device)[None, :] < n_tgt.to(self.device)[:, None])
        tok_lp = torch.gather(logprobs, 2, tgt_dev[:, :, None]).squeeze(-1)
        tok_lp = torch.where(valid, tok_lp, torch.zeros_like(tok_lp))
        lp_sum = tok_lp.sum(dim=1)
        lp_tok = lp_sum / n_tgt.to(self.device)
        argmax = logits.argmax(dim=-1)
        em = ((argmax == tgt_dev) | ~valid).all(dim=1)
        first = logits[:, 0, :]
        gold = torch.gather(first, 1, tgt_dev[:, :1]).squeeze(1)
        masked = first.clone()
        masked.scatter_(1, tgt_dev[:, :1], float("-inf"))
        margin = gold - masked.max(dim=1).values

        residuals = session.stacked() if capture else None
        return ForwardResult(
            logprob_sum=lp_sum.cpu().numpy().astype(np.float64),
            logprob_per_token=lp_tok.cpu().numpy().astype(np.float64),
            exact_match=em.cpu().numpy().astype(bool),
            first_token_margin=margin.cpu().numpy().astype(np.float64),
            n_target_tokens=n_tgt.numpy().astype(np.int64),
            residuals=residuals,
        )


class _HookSession:
    """Per-forward state: applies interventions and gathers residuals at the query token."""

    def __init__(
        self,
        n_layers: int,
        query_pos: torch.Tensor,
        interventions: Sequence[Intervention],
        capture: bool,
        device: torch.device,
    ) -> None:
        self.n_layers = n_layers
        self.query_pos = query_pos
        self.batch_idx = torch.arange(query_pos.shape[0], device=device)
        self.capture = capture
        self.captured: dict[int, torch.Tensor] = {}
        self.by_layer: dict[int, list[Intervention]] = {}
        for iv in interventions:
            if not (0 <= iv.layer <= n_layers):
                raise ValueError(f"intervention layer {iv.layer} outside 0..{n_layers}")
            self.by_layer.setdefault(iv.layer, []).append(iv)
        self.device = device

    def process(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        ivs = self.by_layer.get(layer, ())
        if ivs:
            hidden = hidden.clone()
            current = hidden[self.batch_idx, self.query_pos].float()
            for iv in ivs:
                vec = torch.as_tensor(np.asarray(iv.vectors, dtype=np.float32), device=self.device)
                if vec.ndim == 1:
                    vec = vec[None, :]
                current = current + iv.scale * vec
            hidden[self.batch_idx, self.query_pos] = current.to(hidden.dtype)
        if self.capture:
            self.captured[layer] = hidden[self.batch_idx, self.query_pos].detach().float().cpu()
        return hidden

    def stacked(self) -> np.ndarray:
        missing = [l for l in range(self.n_layers + 1) if l not in self.captured]
        if missing:
            raise RuntimeError(f"residuals not captured at layers {missing}")
        return torch.stack([self.captured[l] for l in range(self.n_layers + 1)]).numpy().astype(np.float32)


def _concat_results(parts: list[ForwardResult]) -> ForwardResult:
    return ForwardResult(
        logprob_sum=np.concatenate([p.logprob_sum for p in parts]),
        logprob_per_token=np.concatenate([p.logprob_per_token for p in parts]),
        exact_match=np.concatenate([p.exact_match for p in parts]),
        first_token_margin=np.concatenate([p.first_token_margin for p in parts]),
        n_target_tokens=np.concatenate([p.n_target_tokens for p in parts]),
        residuals=None if parts[0].residuals is None else np.concatenate([p.residuals for p in parts], axis=1),
    )


def environment_metadata() -> dict[str, Any]:
    import importlib.metadata as im

    pkgs = {}
    for name in ("torch", "transformers", "numpy", "matplotlib", "pyyaml", "tokenizers", "safetensors"):
        try:
            pkgs[name] = im.version(name)
        except im.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": pkgs,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hf_home": os.environ.get("HF_HOME"),
    }


def set_torch_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
