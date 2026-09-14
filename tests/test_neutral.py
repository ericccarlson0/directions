"""The neutral prose prompts of the damage measure (docs/DECISIONS.md D24)."""

from directions.neutral import NEUTRAL_SENTENCES, neutral_prompts


def test_neutral_sentences_are_distinct_prose_cut_mid_sentence():
    assert len(NEUTRAL_SENTENCES) == 48 and len(set(NEUTRAL_SENTENCES)) == 48
    for s in NEUTRAL_SENTENCES:
        assert not s.endswith((".", "?", "!")) and "Q:" not in s and "\n" not in s and len(s.split()) >= 8
    ps = neutral_prompts()
    assert [p.prompt for p in ps] == list(NEUTRAL_SENTENCES) and all(p.target == " the" and p.demos == () for p in ps)
    assert len(neutral_prompts(5)) == 5
