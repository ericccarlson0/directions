"""Neutral prose prompts for the damage measure (docs/DECISIONS.md D24).

Steering a task prompt is supposed to change the next-token distribution at the
query token, so a divergence measured there mixes the intended change with
collateral damage. These sentences carry no task: the intervention is applied at
their last token and the next-token distribution is compared with the unsteered
one. A direction that disturbs prose continuation at the strength it is injected
with has stopped being a perturbation of the stream.
"""

from __future__ import annotations

from .prompts import Prompt
from .tasks import Item

# Cut mid-sentence so that a continuation is expected; varied register and topic; no lists, no questions.
NEUTRAL_SENTENCES: tuple[str, ...] = (
    "The train left the station a few minutes late, and by the time it reached the coast the",
    "She had always believed that the best part of any journey was the",
    "After the rain stopped, the streets of the old town filled with people who",
    "The committee met on Thursday to discuss the budget for the",
    "He opened the letter slowly, unsure whether the news inside would be",
    "In the early morning light, the mountains looked closer than they",
    "The recipe calls for two cups of flour, a pinch of salt, and",
    "Most of the houses on the street were built in the years after the",
    "When the museum reopened, visitors were surprised to find that the",
    "The engineer explained that the bridge had been designed to withstand",
    "Nobody in the village could remember a winter as cold as the",
    "The children spent the afternoon building a fort out of",
    "Her research focused on the way rivers change course over",
    "The orchestra tuned their instruments while the audience settled into",
    "By the end of the season, the farmers had harvested more wheat than",
    "The old clock in the hallway had not worked properly since the",
    "He parked the car under a tree and walked the rest of the way to the",
    "The library extended its opening hours because so many students",
    "A light breeze came in through the window and scattered the papers on the",
    "The doctor recommended rest, plenty of water, and a return visit in",
    "Their conversation drifted from the weather to politics and finally to the",
    "The lighthouse had guided ships past the rocks for more than a",
    "The new software update fixed several bugs but introduced a problem with the",
    "At the market, the price of tomatoes had risen sharply because of the",
    "The painter mixed the colours carefully, trying to match the shade of the",
    "The team celebrated the victory with a dinner at the restaurant near the",
    "The manuscript was discovered in a drawer in the attic of a house in",
    "Because the road was closed, the bus took a longer route through the",
    "The lecture covered the history of the printing press and its effect on the",
    "The garden needed watering every evening during the long, dry",
    "The pilot announced that the flight would land about twenty minutes",
    "Every summer the family rented the same small cottage by the",
    "The bakery on the corner sells out of bread before nine o'clock most",
    "The astronomer pointed the telescope at a faint object near the edge of the",
    "The contract was signed after months of negotiation between the two",
    "The dog waited by the door every afternoon until the children came home from",
    "The archaeologists uncovered a mosaic floor beneath the ruins of the",
    "The river froze so thickly that people walked across it to reach the",
    "The tailor measured the sleeve twice before cutting the",
    "The election results were announced shortly after midnight, and the",
    "The hikers reached the summit just as the fog began to",
    "The company moved its headquarters to a larger building on the other side of the",
    "The novel begins in a small coastal town where the narrator has just",
    "The nurse checked the chart and adjusted the dose of the",
    "The choir rehearsed the final piece one more time before the",
    "The ferry crosses the strait four times a day except during the",
    "The professor's office was lined with books on every subject from",
    "The storm knocked out power to the whole valley for nearly two",
)

_TARGET = " the"  # a dummy continuation so the prompt can be scored; only the query-token distribution is used


def neutral_prompts(n: int | None = None) -> list[Prompt]:
    """The neutral sentences as prompts (the intervention acts at their last token)."""
    texts = NEUTRAL_SENTENCES if n is None else NEUTRAL_SENTENCES[:n]
    return [Prompt(prompt=t, target=_TARGET, query=Item(input=t, output=_TARGET.strip()), demos=()) for t in texts]
