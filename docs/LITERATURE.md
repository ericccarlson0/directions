
# Related Work

## Small Vectors, Big Effects: A Mechanistic Study of RL-Induced Reasoning via Steering Vectors (2026)
	"follow a control through depth"
	originally attempting to understand finetuning (RL)
	(https://arxiv.org/html/2509.06608v4)

## Verbalizable Representations Form a Global Workspace in Language Models (2026)
	... figures out a version of the amplification question
	defines a per-layer MLP gain
	(https://transformer-circuits.pub/2026/workspace/)

## Activation Transport Operators (2025)
	... about the transformation of directions in the residual stream
	(https://arxiv.org/abs/2508.17540)

## Steerable but not Decodable: Function Vectors Act Beyond the Logit Lens (2026)
	... about the measurability of function vectors
	(https://arxiv.org/abs/2604.02608)

## The Refusal Axis is Layer Local (2026)
	... about the transformation of the direction (but not function/control vectors; only one steering vector)
	(https://www.jordanbrowne-moore.com/articles/activation-steering)

## As Language Models Scale, Low-order Linear Depth Dynamics Emerge
	(https://arxiv.org/abs/2603.12541)
	a "systems-theoretic foundation" for analyzing and controlling dynamics
	"[the paper] directly predicts and validates layerwise steering sensitivity"

# The variable-map line (D45–D55): prior work

Checked 2026-10-19 against the papers' text (abstract only where marked). Our findings: (1) operands stay at
their digit tokens until a gather at 0.67–0.75 of depth, where the sums arrive at the answer token, never
assembled at `)`, `-` or `=`; (2) the sums are decodable there but rarely held as separate causal variables (DAS,
k ≤ 64; two brackets 0/1/2/0 for the 1.7B/4B/8B/14B, three brackets and max 0); (3) the answer becomes a variable
at or just after the gather; (4) max's comparison bit is decodable early but not a clean variable; (5) no single
head carries a sum.

## Anticipated (replication or the same conclusion by other means)

## A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis (2023)
	Stolfo, Belinkov, Sachan, EMNLP. Single operations (GPT-J and others): last-token attention mid-stack moves the
	operands, late MLPs (about 0.7 of depth) write the result. Anticipates (1) for one operation.
	(https://arxiv.org/abs/2305.15054)

## Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics (2025)
	Nikankin et al., ICLR. "a op b" in Llama3-8B and others: a few heads copy operands to the last token, MLP
	neurons apply operand-pattern heuristics. A mechanism for (2): no need to hold a sum.
	(https://arxiv.org/abs/2410.21272)

## Language Models Use Trigonometry to Do Addition (2025); Interpreting and Improving LLMs in Arithmetic Calculation (2024)
	Kantamneni & Tegmark; Zhang et al., ICML. Single additions: operand transport by heads, then MLPs compute.
	(https://arxiv.org/abs/2502.00873) (https://arxiv.org/abs/2409.01659)

## Do Language Models Use Their Depth Efficiently? (2025)
	Csordás, Manning, Potts, NeurIPS. Llama 3.1 and Qwen 3 (secondary): input tokens matter until mid-depth, and
	"later hops of computation do not use more depth, indicating that no composition is happening". The closest
	prior conclusion to (2), from residual erasure and integrated gradients on examples, not variable-level tests.
	(https://arxiv.org/abs/2505.13898)

## MIB: A Mechanistic Interpretability Benchmark (2025)
	Mueller et al., ICML. Its arithmetic task scores DAS and other methods on one intermediate variable,
	carry-the-one: "Gemma-2 results are at chance", Llama-3.1 "faint signs of success". The same kind of null as
	(2), on a different variable.
	(https://arxiv.org/abs/2504.13151)

## Implicit Reasoning in Transformers is Reasoning through Shortcuts (2025)
	Lin et al., Findings ACL. Chained modular addition and subtraction: without a fixed premise order, models chain
	numbers directly instead of computing steps, and fail when a variable is the subtrahend; large LLMs degrade the
	same way. A shortcut account of (2): (a+b)−(c+d) flattens to a+b−c−d; max does not flatten.
	(https://arxiv.org/abs/2503.07604)

## Interpretability Analysis of Arithmetic In-Context Learning in LLMs (2025)
	Polyakov et al., EMNLP. Pythia-12B: partial sums readable by the logit lens but not used. Decodable but unused,
	as in (2). Probing critiques: Elazar et al. 2021 (https://arxiv.org/abs/2006.00995), Ravichander et al. 2021.
	(https://aclanthology.org/2025.emnlp-main.92/)

## In tension

## States Hidden in Hidden States: Implicit Discrete State Representations Emerge in LLMs' Hidden States (2024)
	Chen, Hu, Liu, Sun. Chained addition (13+24+41=) in Qwen 4B–72B and DeepSeek: MLP probes at the operator tokens
	decode the running sum, and an attention intervention suggests it is used. Against (1), on a serial task with
	multi-digit operands and an earlier Qwen; our swaps say the `)` and `-` states are not needed, not that nothing
	is computed there.
	(https://arxiv.org/abs/2407.11421)

## On the Biology of a Large Language Model (2025)
	Lindsey et al. Claude 3.5 Haiku: for "(4 + 5) * 3" a feature for the intermediate 9; addition runs a coarse
	magnitude path beside an exact last-digit path (cf. max's accuracy rising with the gap between the sums).
	Attribution graphs on single prompts of a much larger model.
	(https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

## Interpretability at Scale: Identifying Causal Mechanisms in Alpaca (2023)
	Wu, Geiger, Icard, Potts, Goodman, NeurIPS (Boundless DAS). Alpaca-7B, "between X and Y": boolean boundary
	variables found with IIA 0.86–0.90, extracted right after the price and copied to the last token, each using
	about 200–400 of 4096 dimensions. Against (4); their comparands are given, not computed, and their subspaces are
	far larger than our k ≤ 64.
	(https://arxiv.org/abs/2305.08809)

## Method
	DAS: Geiger et al. 2024 (https://arxiv.org/abs/2303.02536); causal abstraction, JMLR 2025
	(https://jmlr.org/papers/v26/23-0058.html); RAVEL's Cause/Isolate (https://arxiv.org/abs/2402.17700); the
	subspace-patching illusion, Makelov et al. 2024 (https://arxiv.org/abs/2311.17030); the non-linear
	representation dilemma, Sutter et al. 2025 (https://arxiv.org/abs/2507.08802): our nulls are nulls for linear
	encodings only.

## Not found
	No mechanistic or DAS study of (a+b)−(c+d), three brackets or max(a+b, c+d) in pretrained LMs, and none that
	counts how many computed intermediates a model holds as separate variables, or how that count changes with size.
