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

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids)


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

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


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
class HeadPatch:
    """Replace one attention head's output at each example's query token in block ``layer``.

    The patch applies to the head's slice of the input of the attention output projection
    (``o_proj``), i.e. the head's output before it is mixed into the residual stream; this is
    the activation Todd et al. (2024) average and patch for function vectors. ``heads`` is one
    head index shared by all examples or one per example, shape ``(N,)``; ``values`` is one
    vector of shape ``(head_dim,)`` or one per example, shape ``(N, head_dim)``.
    """

    layer: int
    heads: np.ndarray | int
    values: np.ndarray


@dataclass
class ForwardResult:
    logprob_sum: np.ndarray  # (N,) teacher-forced log-probability of the target
    logprob_per_token: np.ndarray  # (N,) logprob_sum / n_target_tokens
    exact_match: np.ndarray  # (N,) bool: argmax equals target at every target position
    first_token_margin: np.ndarray  # (N,) gold logit minus best competing logit, first target token
    n_target_tokens: np.ndarray  # (N,)
    residuals: np.ndarray | None  # (L+1, N, d) float32 at the final query token, if captured
    head_outputs: np.ndarray | None = None  # (L, N, n_heads, head_dim) float32 at the query token, if captured
    first_token_logprob: np.ndarray | None = None  # (N,) log p of the first target token (D21 head ranking)
    query_logprobs: np.ndarray | None = None  # (N, V) float32 next-token log-probabilities at the query token, if captured
    # Damage against a reference run's query-token distribution (docs/DECISIONS.md D24), if one was given:
    kl_from_reference: np.ndarray | None = None  # (N,) KL(reference || this run) at the query token
    collateral_kl: np.ndarray | None = None  # (N,) the same KL with the first target token removed from both (renormalised)
    argmax_changed: np.ndarray | None = None  # (N,) bool: the most likely next token differs from the reference's

    def metrics_dict(self) -> dict[str, float]:
        out = {
            "logprob_per_token_mean": float(np.mean(self.logprob_per_token)),
            "logprob_sum_mean": float(np.mean(self.logprob_sum)),
            "accuracy": float(np.mean(self.exact_match)),
            "first_token_margin_mean": float(np.mean(self.first_token_margin)),
            "n": int(self.logprob_sum.shape[0]),
        }
        if self.kl_from_reference is not None:
            out["kl_mean"] = float(np.mean(self.kl_from_reference))
            out["kl_median"] = float(np.median(self.kl_from_reference))
            out["collateral_kl_mean"] = float(np.mean(self.collateral_kl))
            out["collateral_kl_median"] = float(np.median(self.collateral_kl))
            out["argmax_change_rate"] = float(np.mean(self.argmax_changed))
        return out


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
        # examples/batches through run() and gradients(); read per stage by directions.profiling.Profiler
        self.counters: dict[str, int] = {"forward_examples": 0, "forward_batches": 0, "gradient_examples": 0, "gradient_batches": 0}
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
        self.n_heads = int(self.model.config.num_attention_heads)
        self.head_dim = int(getattr(self.model.config, "head_dim", self.hidden_size // self.n_heads))
        self.attn_out = [self._find_attn_out(b) for b in self.blocks]
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

    @staticmethod
    def _find_attn_out(block: torch.nn.Module) -> torch.nn.Linear:
        """The attention output projection of a block (``self_attn.o_proj`` on Qwen/Llama-style models)."""
        attn = getattr(block, "self_attn", None)
        proj = getattr(attn, "o_proj", None) if attn is not None else None
        if not isinstance(proj, torch.nn.Linear):
            raise ValueError("could not locate the attention output projection (self_attn.o_proj) on this model")
        return proj

    def head_to_residual(self, layer: int, head: int, z: np.ndarray) -> np.ndarray:
        """The residual-stream vector head ``head`` of block ``layer`` writes for its output ``z`` (``head_dim``,).

        The output projection is linear (no bias on the supported models), so the attention output is the
        sum of these per-head terms; a function vector is the sum over its selected heads. float64.
        """
        proj = self.attn_out[layer]
        if proj.bias is not None:
            raise ValueError("attention output projection with a bias is not supported")
        W = proj.weight.detach().to(torch.float64).cpu().numpy()  # (d, n_heads * head_dim)
        s = slice(head * self.head_dim, (head + 1) * self.head_dim)
        return W[:, s] @ np.asarray(z, dtype=np.float64)

    def metadata(self) -> dict[str, Any]:
        n_params = int(sum(p.numel() for p in self.model.parameters()))
        meta = {
            "backend": self.cfg.backend,
            "name": self.cfg.name if self.cfg.backend == "hf" else f"toy:{self.cfg.toy}",
            "architecture": type(self.model).__name__,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
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
            self.attn_out[l].register_forward_pre_hook(self._make_head_hook(l))
        self.blocks[L - 1].register_forward_hook(self._make_post_hook(L))

    def _make_head_hook(self, layer: int):
        def hook(module, args):
            if self._session is None:
                return None
            x = args[0]
            new = self._session.process_heads(layer, x)
            return None if new is x else (new,) + tuple(args[1:])

        return hook

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

    def target_tokens(self, target_text: str) -> list[str]:
        """The target's tokens as text pieces (the way the target is scored)."""
        ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        return [self.tokenizer.decode([i]) for i in ids]

    def scored_token_count(self, target_text: str, score_reference: str | None) -> int:
        """How many of the target's tokens its log-probability counts (all, or the changed ones; D30)."""
        ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        return int(sum(self._scored_tokens(ids, score_reference)))

    def first_target_token_ids(self, prompts: Sequence[Prompt]) -> np.ndarray:
        """The id of each prompt's first target token (the token scored at the query position), ``(N,)``."""
        return np.array([self._encode_prompt(p)[1][0] for p in prompts], dtype=np.int64)

    def unembedding_directions(self, token_ids: np.ndarray) -> np.ndarray:
        """The residual-stream directions that raise the logits of ``token_ids``, ``(N, d)`` float64.

        The logit of token ``t`` is ``W_U[t] · norm(h)``; on the supported models the final norm is an RMSNorm
        with a per-dimension scale ``g``, so up to the (positive) normalisation factor the direction in the
        residual stream is ``g * W_U[t]``. Without a scaled final norm it is ``W_U[t]`` itself. Not normalised.
        """
        ids = torch.as_tensor(np.asarray(token_ids, dtype=np.int64), device=self.device)
        rows = self.model.lm_head.weight.detach()[ids].to(torch.float64)  # (N, d)
        norm = getattr(getattr(self.model, "model", None), "norm", None)
        gain = getattr(norm, "weight", None)
        if gain is not None and gain.shape == rows.shape[1:]:
            rows = rows * gain.detach().to(torch.float64)[None, :]
        return rows.cpu().numpy()

    def logits_from_residual(self, h: Any) -> Any:
        """The next-token logits the model would produce from final-block residuals ``h`` (n, d): the final
        norm and the unembedding, evaluated in float32 (the logit lens of a residual-stream vector; D33).
        ``h`` is a tensor on the model's device or an array; returns a float32 tensor (n, V) on the device."""
        x = torch.as_tensor(np.asarray(h) if not isinstance(h, torch.Tensor) else h, device=self.device).to(torch.float32)
        norm = getattr(getattr(self.model, "model", None), "norm", None)
        if norm is not None and getattr(norm, "weight", None) is not None:
            eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm.weight.detach().to(torch.float32)
        W = self.model.lm_head.weight.detach().to(torch.float32)
        with torch.inference_mode():
            return torch.nn.functional.linear(x, W)

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
        head_patches: Sequence[HeadPatch] = (),
        capture_heads: bool = False,
        capture_logprobs: bool = False,
        reference_logprobs: Any = None,
    ) -> ForwardResult:
        """Teacher-forced forward over ``prompts`` with optional interventions/patches/capture.

        ``capture_logprobs`` keeps the full next-token log-probabilities at the query token (``(N, V)``);
        ``reference_logprobs`` (the same array from an unsteered run of the same prompts, as returned by
        :meth:`reference_tensor` or as a NumPy array) makes the run record the KL divergence from that
        distribution and whether the most likely token changed (the damage check, D24).
        """
        bs = batch_size or self.cfg.batch_size
        outs: list[ForwardResult] = []
        if reference_logprobs is not None and not isinstance(reference_logprobs, torch.Tensor):
            reference_logprobs = self.reference_tensor(reference_logprobs)
        if reference_logprobs is not None and reference_logprobs.shape[0] != len(prompts):
            raise ValueError("reference_logprobs must have one row per prompt")
        for start in range(0, len(prompts), bs):
            batch = prompts[start : start + bs]
            self.counters["forward_examples"] += len(batch)
            self.counters["forward_batches"] += 1
            sliced = [
                Intervention(iv.layer, iv.vectors if iv.vectors.ndim == 1 else iv.vectors[start : start + bs], iv.scale)
                for iv in interventions
            ]
            patches = [
                HeadPatch(
                    hp.layer,
                    hp.heads if np.ndim(hp.heads) == 0 else np.asarray(hp.heads)[start : start + bs],
                    hp.values if np.ndim(hp.values) == 1 else np.asarray(hp.values)[start : start + bs],
                )
                for hp in head_patches
            ]
            ref = None if reference_logprobs is None else reference_logprobs[start : start + bs]
            outs.append(self._run_batch(batch, sliced, capture, patches, capture_heads, capture_logprobs, ref))
        return _concat_results(outs)

    def reference_tensor(self, query_logprobs: np.ndarray) -> torch.Tensor:
        """``query_logprobs`` of a captured run as a float32 tensor on the model's device (kept for several runs)."""
        return torch.as_tensor(np.asarray(query_logprobs, dtype=np.float32), device=self.device)

    def gradients(self, prompts: Sequence[Prompt], batch_size: int | None = None,
                  interventions: Sequence[Intervention] = (), read_points: Sequence[int] | None = None) -> np.ndarray:
        """Gradient of the summed (scored) target log-probability with respect to the residual
        stream at every read point ``0..L`` (or at ``read_points`` only), taken at each example's
        final query token.

        Returns an array shaped ``(n_read_points, N, d)`` (float32). Model parameters are
        frozen, so the backward pass only produces activation gradients; the forward pass is the
        teacher-forced pass used everywhere else, with ``interventions`` applied (none by default).
        With an additive intervention at a read point, the gradient there is also the gradient
        with respect to the added vector (docs/DECISIONS.md D31).
        """
        return self.gradients_with_scores(prompts, batch_size, interventions, read_points)[0]

    def gradients_with_scores(self, prompts: Sequence[Prompt], batch_size: int | None = None,
                              interventions: Sequence[Intervention] = (), read_points: Sequence[int] | None = None
                              ) -> tuple[np.ndarray, np.ndarray]:
        """``gradients`` plus the summed scored log-probability per example, ``(N,)`` float64, from the same pass."""
        bs = batch_size or self.cfg.batch_size
        parts = []
        scores = []
        for start in range(0, len(prompts), bs):
            batch = prompts[start : start + bs]
            self.counters["gradient_examples"] += len(batch)
            self.counters["gradient_batches"] += 1
            g, s = self._gradient_batch(batch, interventions, read_points)
            parts.append(g)
            scores.append(s)
        return np.concatenate(parts, axis=1), np.concatenate(scores)

    def _prepare_batch(self, prompts: Sequence[Prompt]) -> dict[str, Any]:
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
        scored = torch.zeros((B, Tt), dtype=torch.bool)
        for i, ((a, b), p) in enumerate(zip(encoded, prompts)):
            ids = a + b
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attn[i, : len(ids)] = 1
            query_pos[i] = len(a) - 1
            target_ids[i, : len(b)] = torch.tensor(b)
            scored[i, : len(b)] = torch.tensor(self._scored_tokens(b, p.score_reference))
        return {
            "input_ids": input_ids.to(self.device),
            "attn": attn.to(self.device),
            "query_pos": query_pos.to(self.device),
            "target_ids": target_ids.to(self.device),
            "scored": scored.to(self.device),
            "n_tgt": n_tgt,
            "T": T,
            "Tt": Tt,
        }

    def _scored_tokens(self, target_ids: list[int], score_reference: str | None) -> list[bool]:
        """Which target tokens count towards the target's log-probability (docs/DECISIONS.md D30).

        All of them, unless the prompt carries a ``score_reference`` (the input rendered as a target): then a
        token is unchanged, and not scored, when the reference has the same token at the same position under
        the left alignment (a shared leading space) or under the right alignment (the digits of a number an
        operation leaves untouched); the rest are scored. A target identical to its reference keeps every token.
        """
        if score_reference is None:
            return [True] * len(target_ids)
        ref = self.tokenizer.encode(score_reference, add_special_tokens=False)
        shift = len(target_ids) - len(ref)

        def unchanged(j: int, t: int) -> bool:
            left = j < len(ref) and ref[j] == t
            right = 0 <= j - shift < len(ref) and ref[j - shift] == t
            return left or right

        mask = [not unchanged(j, t) for j, t in enumerate(target_ids)]
        return mask if any(mask) else [True] * len(target_ids)

    def _forward_scores(self, batch: dict[str, Any], session: _HookSession, capture_logprobs: bool = False,
                        reference: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Teacher-forced forward with ``session`` installed; target-position scores."""
        T, Tt = batch["T"], batch["Tt"]
        query_pos_dev = batch["query_pos"]
        n_tgt_dev = batch["n_tgt"].to(self.device)
        self._session = session
        try:
            base_out = self.model.model(input_ids=batch["input_ids"], attention_mask=batch["attn"])
        finally:
            self._session = None
        hidden = base_out.last_hidden_state  # (B, T, d) after final norm

        # positions whose logits predict target token j: query_pos + j
        pos = (query_pos_dev[:, None] + torch.arange(Tt, device=self.device)[None, :]).clamp(max=T - 1)
        gathered = torch.gather(hidden, 1, pos[:, :, None].expand(-1, -1, hidden.shape[-1]))
        logits = self.model.lm_head(gathered).float()  # (B, Tt, V)
        logprobs = torch.log_softmax(logits, dim=-1)
        tgt_dev = batch["target_ids"]
        valid = torch.arange(Tt, device=self.device)[None, :] < n_tgt_dev[:, None]
        scored = batch["scored"]  # subset of valid: the tokens that count (D30; all of them by default)
        tok_lp = torch.gather(logprobs, 2, tgt_dev[:, :, None]).squeeze(-1)
        tok_lp = torch.where(valid, tok_lp, torch.zeros_like(tok_lp))
        lp_sum = torch.where(scored, tok_lp, torch.zeros_like(tok_lp)).sum(dim=1)
        lp_tok = lp_sum / scored.sum(dim=1)
        argmax = logits.argmax(dim=-1)
        em = ((argmax == tgt_dev) | ~valid).all(dim=1)
        first = logits[:, 0, :]
        gold = torch.gather(first, 1, tgt_dev[:, :1]).squeeze(1)
        masked = first.detach().clone()
        masked.scatter_(1, tgt_dev[:, :1], float("-inf"))
        margin = gold - masked.max(dim=1).values
        out = {"lp_sum": lp_sum, "lp_tok": lp_tok, "em": em, "margin": margin, "first_lp": tok_lp[:, 0]}
        query_lp = logprobs[:, 0, :]  # the next-token distribution at the query token
        if capture_logprobs:
            out["query_logprobs"] = query_lp
        if reference is not None:
            ref = reference.to(query_lp.dtype)
            out["kl"] = (ref.exp() * (ref - query_lp)).sum(dim=1)
            out["argmax_changed"] = query_lp.argmax(dim=1) != ref.argmax(dim=1)
            # collateral change (D24): the same divergence with the correct first target token removed from both
            # distributions and the rest renormalised, so mass moved onto the answer itself does not count
            first = tgt_dev[:, :1]
            neg_inf = torch.full_like(query_lp[:, :1], float("-inf"))
            ref_rest = torch.log_softmax(ref.scatter(1, first, neg_inf), dim=1)
            cur_rest = torch.log_softmax(query_lp.scatter(1, first, neg_inf), dim=1)
            out["collateral_kl"] = torch.where(torch.isinf(ref_rest), torch.zeros_like(ref_rest),
                                               ref_rest.exp() * (ref_rest - cur_rest)).sum(dim=1)
        return out

    def _run_batch(
        self,
        prompts: Sequence[Prompt],
        interventions: Sequence[Intervention],
        capture: bool,
        head_patches: Sequence[HeadPatch] = (),
        capture_heads: bool = False,
        capture_logprobs: bool = False,
        reference: torch.Tensor | None = None,
    ) -> ForwardResult:
        batch = self._prepare_batch(prompts)
        session = _HookSession(
            n_layers=self.n_layers,
            query_pos=batch["query_pos"],
            interventions=interventions,
            capture=capture,
            device=self.device,
            head_patches=head_patches,
            capture_heads=capture_heads,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
        )
        sc = self._forward_scores(batch, session, capture_logprobs, reference)
        residuals = session.stacked() if capture else None
        heads = session.stacked_heads() if capture_heads else None
        return ForwardResult(
            logprob_sum=sc["lp_sum"].cpu().numpy().astype(np.float64),
            logprob_per_token=sc["lp_tok"].cpu().numpy().astype(np.float64),
            exact_match=sc["em"].cpu().numpy().astype(bool),
            first_token_margin=sc["margin"].cpu().numpy().astype(np.float64),
            n_target_tokens=batch["n_tgt"].numpy().astype(np.int64),
            residuals=residuals,
            head_outputs=heads,
            first_token_logprob=sc["first_lp"].cpu().numpy().astype(np.float64),
            query_logprobs=sc["query_logprobs"].cpu().numpy().astype(np.float32) if capture_logprobs else None,
            kl_from_reference=sc["kl"].cpu().numpy().astype(np.float64) if reference is not None else None,
            collateral_kl=sc["collateral_kl"].cpu().numpy().astype(np.float64) if reference is not None else None,
            argmax_changed=sc["argmax_changed"].cpu().numpy().astype(bool) if reference is not None else None,
        )

    def _gradient_batch(self, prompts: Sequence[Prompt], interventions: Sequence[Intervention] = (),
                        read_points: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        batch = self._prepare_batch(prompts)
        session = _HookSession(
            n_layers=self.n_layers,
            query_pos=batch["query_pos"],
            interventions=interventions,
            capture=False,
            device=self.device,
            grad_capture=True,
        )
        with torch.enable_grad():
            sc = self._forward_scores(batch, session)
            sc["lp_sum"].sum().backward()
        return session.stacked_gradients(read_points), sc["lp_sum"].detach().cpu().numpy().astype(np.float64)


class _HookSession:
    """Per-forward state: applies interventions and gathers residuals at the query token."""

    def __init__(
        self,
        n_layers: int,
        query_pos: torch.Tensor,
        interventions: Sequence[Intervention],
        capture: bool,
        device: torch.device,
        grad_capture: bool = False,
        head_patches: Sequence[HeadPatch] = (),
        capture_heads: bool = False,
        n_heads: int = 0,
        head_dim: int = 0,
    ) -> None:
        self.n_layers = n_layers
        self.query_pos = query_pos
        self.batch_idx = torch.arange(query_pos.shape[0], device=device)
        self.capture = capture
        self.captured: dict[int, torch.Tensor] = {}
        self.grad_capture = grad_capture
        self.grad_tensors: dict[int, torch.Tensor] = {}
        self.by_layer: dict[int, list[Intervention]] = {}
        for iv in interventions:
            if not (0 <= iv.layer <= n_layers):
                raise ValueError(f"intervention layer {iv.layer} outside 0..{n_layers}")
            self.by_layer.setdefault(iv.layer, []).append(iv)
        self.device = device
        self.n_heads, self.head_dim = n_heads, head_dim
        self.capture_heads = capture_heads
        self.captured_heads: dict[int, torch.Tensor] = {}
        self.patches_by_layer: dict[int, list[HeadPatch]] = {}
        for hp in head_patches:
            if not (0 <= hp.layer < n_layers):
                raise ValueError(f"head patch layer {hp.layer} outside 0..{n_layers - 1}")
            self.patches_by_layer.setdefault(hp.layer, []).append(hp)

    def process_heads(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """Patch/capture per-head outputs (the input of the attention output projection) at the query token."""
        patches = self.patches_by_layer.get(layer, ())
        B = self.batch_idx.shape[0]
        if patches:
            x = x.clone()
            cur = x[self.batch_idx, self.query_pos].view(B, self.n_heads, self.head_dim).clone()
            for hp in patches:
                heads = torch.as_tensor(np.broadcast_to(np.asarray(hp.heads, dtype=np.int64), (B,)).copy(), device=self.device)
                vals = np.asarray(hp.values, dtype=np.float32)
                vals = np.broadcast_to(vals, (B, self.head_dim)).copy()
                cur[self.batch_idx, heads] = torch.as_tensor(vals, device=self.device).to(cur.dtype)
            x[self.batch_idx, self.query_pos] = cur.view(B, self.n_heads * self.head_dim)
        if self.capture_heads:
            self.captured_heads[layer] = (
                x[self.batch_idx, self.query_pos].detach().float().cpu().view(B, self.n_heads, self.head_dim)
            )
        return x

    def stacked_heads(self) -> np.ndarray:
        """Per-head outputs at the query token, ``(L, B, n_heads, head_dim)`` float32."""
        missing = [l for l in range(self.n_layers) if l not in self.captured_heads]
        if missing:
            raise RuntimeError(f"head outputs not captured at layers {missing}")
        return torch.stack([self.captured_heads[l] for l in range(self.n_layers)]).numpy().astype(np.float32)

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
        if self.grad_capture:
            # The first read point (embedding output) does not require grad because the
            # parameters are frozen: make it the graph's leaf. Every later read point is a
            # function of it; keep its gradient after the backward pass.
            if not hidden.requires_grad:
                hidden = hidden.detach().requires_grad_(True)
            else:
                hidden.retain_grad()
            self.grad_tensors[layer] = hidden
        return hidden

    def stacked_gradients(self, read_points: Sequence[int] | None = None) -> np.ndarray:
        """Gradients at the query token, ``(n_read_points, B, d)`` float32, after ``backward()`` (all read
        points ``0..L`` by default)."""
        points = list(range(self.n_layers + 1)) if read_points is None else list(read_points)
        missing = [l for l in points if l not in self.grad_tensors or self.grad_tensors[l].grad is None]
        if missing:
            raise RuntimeError(f"gradients not available at layers {missing}")
        grads = [self.grad_tensors[l].grad[self.batch_idx, self.query_pos].detach().float().cpu() for l in points]
        return torch.stack(grads).numpy().astype(np.float32)

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
        head_outputs=None if parts[0].head_outputs is None else np.concatenate([p.head_outputs for p in parts], axis=1),
        first_token_logprob=None if parts[0].first_token_logprob is None else np.concatenate([p.first_token_logprob for p in parts]),
        query_logprobs=None if parts[0].query_logprobs is None else np.concatenate([p.query_logprobs for p in parts]),
        kl_from_reference=None if parts[0].kl_from_reference is None else np.concatenate([p.kl_from_reference for p in parts]),
        collateral_kl=None if parts[0].collateral_kl is None else np.concatenate([p.collateral_kl for p in parts]),
        argmax_changed=None if parts[0].argmax_changed is None else np.concatenate([p.argmax_changed for p in parts]),
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
