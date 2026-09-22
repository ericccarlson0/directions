# Lens-quality prompt sets of the workspace paper

The six `lens-eval-*.json` files are copied unchanged from the companion repository of "Verbalizable
Representations Form a Global Workspace in Language Models" (Anthropic, Transformer Circuits, July 2026;
`anthropics/jacobian-lens`, `data/evaluations/`, commit "Initial release"), under its Apache-2.0 licence
(`LICENSE` beside them). Each file holds prompts (`items[*].prompt`) and the concept words the paper expects a
lens to read at one position (`items[*].intermediates`); `target`, where present, marks the readout position
only and is not scored.

They serve the landmark test (docs/DECISIONS.md D37, `scripts/landmarks.py --lens-file`): on the one
checkpoint with a published Jacobian lens (`Qwen/Qwen3-8B`), the per-read-point rate at which the lens reads
the intermediates locates the paper's workspace band in that model, with the paper's own prompts and its own
instrument. Readout positions follow the paper's README: the final prompt token for multihop, multilingual,
order-ops, association and typo (the token before `target`, or the closing period, or the misspelling's last
fragment), and the last newline (the end of line 1) for poetry; the prompt is truncated there, which leaves the
residual at that position unchanged in a causal model. Ranks are the minimum over the single-token forms of a
word (with and without a leading space; digits and number words, symbols and words for order-ops).
