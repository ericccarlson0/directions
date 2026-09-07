"""PyTorch-hook machinery for residual-stream capture and intervention.

Residual indexing convention (used everywhere in this project)
--------------------------------------------------------------
For a model with ``L`` transformer blocks there are ``L + 1`` residual-stream
read points::

    resid[0]  = embedding output            (input to block 0)
    resid[l]  = output of block l-1         (input to block l),   1 <= l <= L-1
    resid[L]  = output of block L-1         (pre final-norm)

Block ``l`` maps ``resid[l] -> resid[l+1]``, so the blockwise response
``b_l = delta_{l+1} - delta_l`` is exactly the change block ``l`` contributes.
An intervention "at layer l" adds a vector to ``resid[l]``, i.e. to the input of
block ``l``; it is therefore only defined for ``0 <= l <= L-1``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


def resolve_blocks(model) -> list[torch.nn.Module]:
    """Locate the list of transformer blocks in a HF causal-LM."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (torch.nn.ModuleList, list)) and len(obj) > 0:
            return list(obj)
    raise RuntimeError(f"could not locate transformer blocks on {type(model).__name__}")


def _hidden_from_call(args, kwargs):
    if args:
        return args[0], "args"
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"], "kwargs"
    raise RuntimeError("could not find hidden_states in decoder-layer call")


def _replace_hidden(args, kwargs, where, new):
    if where == "args":
        return (new,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs["hidden_states"] = new
    return args, kwargs


@dataclass
class Intervention:
    """Additive residual-stream intervention at one layer and one token.

    ``vector`` is unit-norm; the applied perturbation is ``alpha * vector``
    added to ``resid[layer]`` at ``positions[b]`` for each batch element ``b``.
    """

    layer: int
    vector: torch.Tensor  # (d_model,), unit norm
    alpha: float
    positions: torch.Tensor  # (batch,) long


class ResidualHooks:
    """Context manager attaching capture and intervention hooks.

    Captured residuals are gathered at the requested token positions *inside*
    the hook, so full ``(batch, seq, d_model)`` activations are never retained
    and the memory cost is ``O(batch * layers * d_model)``.
    """

    def __init__(
        self,
        blocks: list[torch.nn.Module],
        positions: torch.Tensor | None = None,
        capture: bool = True,
        intervention: Intervention | None = None,
    ):
        self.blocks = blocks
        self.n_layers = len(blocks)
        self.positions = positions
        self.capture = capture
        self.intervention = intervention
        self.residuals: dict[int, torch.Tensor] = {}
        self._handles: list = []
        if intervention is not None and not 0 <= intervention.layer < self.n_layers:
            raise ValueError(
                f"intervention layer {intervention.layer} outside [0, {self.n_layers - 1}]"
            )

    # -- hook bodies ----------------------------------------------------
    def _gather(self, h: torch.Tensor) -> torch.Tensor:
        idx = self.positions.to(h.device)
        rows = torch.arange(h.shape[0], device=h.device)
        return h[rows, idx].detach().to(torch.float32).cpu()

    def _pre_hook(self, layer_idx: int):
        def fn(module, args, kwargs):
            h, where = _hidden_from_call(args, kwargs)
            if self.intervention is not None and self.intervention.layer == layer_idx:
                iv = self.intervention
                delta = (iv.alpha * iv.vector).to(device=h.device, dtype=h.dtype)
                h = h.clone()
                rows = torch.arange(h.shape[0], device=h.device)
                h[rows, iv.positions.to(h.device)] += delta
                args, kwargs = _replace_hidden(args, kwargs, where, h)
            if self.capture:
                self.residuals[layer_idx] = self._gather(h)
            return args, kwargs

        return fn

    def _final_hook(self, module, args, kwargs, output):
        h = output[0] if isinstance(output, tuple) else output
        self.residuals[self.n_layers] = self._gather(h)

    # -- context manager ------------------------------------------------
    def __enter__(self) -> "ResidualHooks":
        if self.capture and self.positions is None:
            raise ValueError("capture requires token positions")
        for i, block in enumerate(self.blocks):
            need_pre = self.capture or (
                self.intervention is not None and self.intervention.layer == i
            )
            if need_pre:
                self._handles.append(
                    block.register_forward_pre_hook(self._pre_hook(i), with_kwargs=True)
                )
        if self.capture:
            self._handles.append(
                self.blocks[-1].register_forward_hook(self._final_hook, with_kwargs=True)
            )
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def stacked(self) -> torch.Tensor:
        """``(n_layers + 1, batch, d_model)`` float32 tensor of captured residuals."""
        if not self.capture:
            raise RuntimeError("hooks were created with capture=False")
        missing = [i for i in range(self.n_layers + 1) if i not in self.residuals]
        if missing:
            raise RuntimeError(f"missing captured residuals for layers {missing}")
        return torch.stack([self.residuals[i] for i in range(self.n_layers + 1)])


@contextlib.contextmanager
def block_output_patch(block: torch.nn.Module, patch_fn):
    """Temporarily post-process one block's output.

    ``patch_fn(hidden_states) -> hidden_states`` runs on the block's output.
    Used by the exploratory block-ablation analysis to remove or restore a
    single block's steering-induced contribution.
    """

    def hook(module, args, kwargs, output):
        if isinstance(output, tuple):
            return (patch_fn(output[0]),) + tuple(output[1:])
        return patch_fn(output)

    handle = block.register_forward_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
