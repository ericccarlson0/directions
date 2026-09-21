"""The verbalisation test's pure parts (D36): the lens index mapping, the transport, the readout bookkeeping."""

import numpy as np
import pytest
import torch

from directions.verbalise import TASK_WORDS, Lens, readout_summary, transport, word_token_ids


def test_lens_maps_read_points_to_block_outputs(tmp_path):
    J = {l: torch.eye(4) * (l + 2) for l in range(3)}  # a 4-layer model: blocks 0..2 fitted, block 3 is the final residual
    torch.save({"J": J, "n_prompts": 7, "source_layers": [0, 1, 2], "d_model": 4}, tmp_path / "lens.pt")
    lens = Lens.load(str(tmp_path / "lens.pt"))
    assert lens.d_model == 4 and lens.n_prompts == 7 and sorted(lens.jacobians) == [0, 1, 2]
    h = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert torch.allclose(transport(lens, h, 1, 4), torch.tensor([2.0, 0, 0, 0]))  # read point 1 = output of block 0
    assert torch.allclose(transport(lens, h, 3, 4), torch.tensor([4.0, 0, 0, 0]))
    assert torch.allclose(transport(lens, h, 4, 4), torch.tensor(h))  # the last read point: identity
    assert torch.allclose(transport(None, h, 2, 4), torch.tensor(h))  # the logit lens: identity everywhere
    with pytest.raises(ValueError):
        lens.matrix_for_read_point(0, 4)
    with pytest.raises(ValueError):
        torch.save({"weights": 1}, tmp_path / "bad.pt") or Lens.load(str(tmp_path / "bad.pt"))


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return {" opposite": [5], " plural": [9, 9], " zzz": []}.get(text, [1])

    def decode(self, ids):
        return f"<{ids[0]}>"


def test_word_ids_and_readout_masses():
    ids = word_token_ids(_Tok(), ["opposite", "plural", "zzz"])
    assert ids == {"opposite": 5, "plural": 9}
    logits = np.full(12, -10.0)
    logits[[5, 9, 3]] = [2.0, 1.0, 3.0]
    lp = logits - np.log(np.exp(logits).sum())
    s = readout_summary(lp, _Tok().decode, 3, ids, np.array([3, 3, 7]))
    assert [t for t, _ in s["top"]] == ["<3>", "<5>", "<9>"]
    assert s["task_best"] == "opposite" and s["task_best_rank"] == 2
    assert s["task_mass"] == pytest.approx(float(np.exp(lp[5]) + np.exp(lp[9])))
    assert s["answer_mass"] == pytest.approx(float(np.exp(lp[3]) + np.exp(lp[7]))) and s["answer_best_rank"] == 1
    assert s["entropy"] > 0
    assert set(TASK_WORDS) >= {"antonym", "plural", "past_tense", "singular", "uppercase", "number_to_words"}
