# LLM model routing: 19 papers — gist, benefits, and how this repo tests them

Every paper here answers one question: which model should handle this query, so you pay large-model prices only when it changes the answer. They differ in *when* the decision is made and *what signal* drives it.

| Family | Decides | Signal | Papers |
| --- | --- | --- | --- |
| Predict-then-select | Before any LLM runs | A learned predictor of each model's success, preference or loss | EmbedLLM, RouteLLM, MetaLLM, OptLLM, Multiple Minds, Hybrid LLM, Predictive Preference, Routoo, Fly-Swat, Zooter, Benchmark routing, Tryage |
| Cascade | After the cheap model answers | Confidence in that answer (logprobs, self-verification, a scorer, user feedback) | AutoMix, Two-tier uncertainty, FrugalGPT, EcoAssistant |
| Ensemble / fusion | After all models answer | Ranking or merging of outputs | LLM-Blender, Merge/Ensemble/Cooperate survey |
| Online learning | Continuously, per request | A student trained on the expensive model's answers | Cache & Distil (MetaLLM's online bandit mode too) |

The bench runs every model on every benchmark query once with vLLM, measures $/token on your GPUs, then replays all routers offline on the same held-out split, so their cost-accuracy curves are directly comparable. Names in `code` are registry names (`routerlab/routers/__init__.py`).

## 1. EmbedLLM (Oct 2024) — [arXiv 2410.02223](https://arxiv.org/abs/2410.02223)
Learns one compact vector per LLM, so routing becomes a cheap dot product between a model vector and a query vector.
- **Gist:** treat the (query × model) correctness matrix like a recommender-system ratings matrix; learn a model embedding per LLM and a projection per query, and predict whether that model answers that query correctly. Route to the highest predicted success.
- **Benefits:** better routing accuracy and lower latency than earlier routers; the embeddings forecast a model's benchmark scores without running it and capture traits such as coding strength; a new model is added by fitting only its vector on a small probe set.
- **How we test it:** `embedllm` — matrix factorisation on the train split's correctness matrix, cost penalty λ swept; `add_model()` measures how many probe questions a new model needs.

## 2. RouteLLM (Jun/Jul 2024) — [arXiv 2406.18665](https://arxiv.org/abs/2406.18665)
A binary router that sends each query to a strong or a weak model; reports over 2× cost reduction without losing quality.
- **Gist:** learn P(strong wins | query) from Chatbot Arena preferences plus labels augmented from benchmarks with gold answers; route to the strong model when that probability exceeds α. Four routers: similarity-weighted Elo, matrix factorisation, BERT classifier, causal-LLM classifier.
- **Benefits:** no labels from your own models beyond preference data; routers transfer when the strong/weak pair is swapped at test time; introduced APGR and CPT metrics.
- **How we test it:** `routellm-mf`, `routellm-sw_ranking`, `routellm-clf`; golden labels from your outcome table; `load_arena_pairs()` adds Arena votes; APGR and CPT(50/80) for every router.

## 3. MetaLLM (Jul 2024) — [arXiv 2407.10834](https://arxiv.org/abs/2407.10834)
Model choice as a contextual multi-armed bandit for classification and multiple-choice tasks.
- **Gist:** each LLM is an arm, the query embedding is the context, the reward is accuracy minus weighted cost; a linear reward model per arm picks the best expected reward.
- **Benefits:** very light; fits the case where you only observe the outcome of the model you called; beats single models on cost and accuracy across OpenAI, Together AI and open models.
- **How we test it:** `metallm-offline` (ridge, full information) and `metallm-online` (LinUCB with bandit feedback only), on the cls+MCQ subset. Live: `router:metallm-online` learns from `/feedback`.

## 4. Merge, Ensemble, and Cooperate — survey (Jul 2024) — [arXiv 2407.06089](https://arxiv.org/abs/2407.06089)
Maps how LLMs can work together: *merge* weights, *ensemble* outputs (before/during/after inference), *cooperate* with different roles (cascades, verifiers, speculative decoding).
- **Benefits:** the map for choosing a lever — merging gives one deployment; after-inference ensembles buy quality at N× cost; routing and cascades buy savings.
- **How we test it:** `collab-vote` (weighted majority over the k best models) and `scripts/merge_models.py` (linear / task-arithmetic merge of same-base fine-tunes; add to the pool and re-collect).

## 5. AutoMix (Oct 2023, rev. Jun 2024) — [arXiv 2310.12963](https://arxiv.org/abs/2310.12963)
The small model checks its own answer; escalate only when the check looks unreliable. Reports over 50% lower compute at comparable performance.
- **Gist:** few-shot self-verification sampled k times gives a noisy confidence; a POMDP meta-verifier treats true quality as hidden and decides whether to trust it.
- **Benefits:** no trained router; works with black-box APIs; corrects over/under-confident verifiers; IBC metric measures gain per dollar against mixing two models.
- **How we test it:** 8 sampled verifications per small model at collection; `automix-threshold` and `automix-pomdp` (one-step belief version); verification cost charged; ΔIBC reported.

## 6. OptLLM (May 2024) — [arXiv 2405.15130](https://arxiv.org/abs/2405.15130)
Assigns a whole batch of queries at once and returns a menu of Pareto-optimal trade-offs; 2.4–49.2% cheaper at the best LLM's accuracy.
- **Gist:** multi-label success predictor with ensemble uncertainty, then destroy-and-rebuild multi-objective optimisation.
- **Benefits:** pick a point matching your budget; built for batch jobs; pessimistic predictions avoid gambling.
- **How we test it:** `optllm` — bootstrap MLP ensemble (mean − κ·std) + local search per budget.

## 7. Uncertainty-based two-tier selection (May 2024) — [arXiv 2405.02134](https://arxiv.org/abs/2405.02134)
The small model's own token probabilities are the signal; beat trained routers and cascades in 25 of 27 setups.
- **Gist:** escalate when the top-1/top-2 probability margin at the answer is small. No training, no extra call.
- **Benefits:** zero training data, zero router latency, nothing to maintain; the baseline learned routers must beat.
- **How we test it:** `two-tier-marginmin` offline; `router:two-tier` live with `logprobs=5`.

## 8. Harnessing the Power of Multiple Minds (May 2024) — [arXiv 2405.00467](https://arxiv.org/abs/2405.00467)
Is per-query routing on reasoning tasks worth it? Promising, but not reliable in every setting.
- **Gist:** classification and clustering routers vs. the best single model and the oracle on GSM8K/MMLU-style tasks.
- **Benefits:** a reality check — the oracle gap is large but learned routers recover only part of it. Run it first.
- **How we test it:** `minds-classifier`, `minds-cluster`; the report prints the oracle gap and the share of queries where the best model fails but another succeeds.

## 9. Hybrid LLM (Apr 2024, ICLR 2024) — [arXiv 2404.14618](https://arxiv.org/abs/2404.14618)
Predicts from the query alone whether the small model will be good enough; up to 40% fewer large-model calls with no quality drop.
- **Gist:** quality gap = small − large quality (BARTScore); DeBERTa router on deterministic, probabilistic or transformed (relaxed by t) labels.
- **Benefits:** decision before generation (no cascade latency); quality knob at test time; transformed labels learn "easy" queries.
- **How we test it:** `hybrid-det/-prob/-trans`; quality = reward-model score; `finetune.py` for the full DeBERTa recipe.

## 10. Predictive Human Preference (Feb 2024) — [Chip Huyen](https://huyenchip.com/2024/02/28/predictive-human-preference.html)
Per-prompt model ranking instead of one global leaderboard, then routing.
- **Gist:** prompt-conditioned Bradley-Terry trained on pairwise votes; use the cheap model when it is predicted to be as good.
- **Benefits:** turns preference data into a router and shows *where* each model is strong.
- **How we test it:** `predictive-pref`; pairs from outcomes (correctness, reward tie-break) and optionally Arena votes; `rank()` gives per-prompt leaderboards.

## 11. Routoo (Jan 2024) — [arXiv 2401.13979](https://arxiv.org/abs/2401.13979)
Performance predictor + budget-aware selector; Mixtral-level MMLU accuracy at one-third lower cost, 75.9% at Mixtral's cost.
- **Gist:** a lightweight LM scores every candidate without running them; a selector picks per query under cost/latency constraints.
- **Benefits:** explicit, controllable quality-vs-cost targets.
- **How we test it:** `routoo` — MLP over query, model embedding and their product; Lagrange multiplier by bisection hits the budget exactly; optional latency cap.

## 12. Fly-Swat or Cannon? (Aug 2023, WSDM 2024) — [arXiv 2308.06077](https://arxiv.org/abs/2308.06077)
Meta-model predicts each LM's success, then an assignment problem under a budget; matches the largest LM at 63% lower cost.
- **Gist:** DistilBERT meta-model → integer linear program maximising expected accuracy under a total-cost cap; also a cost-sensitive rule.
- **Benefits:** the budget is a hard guarantee across a batch.
- **How we test it:** `flyswat-ilp` (exact ILP, SciPy HiGHS) and `flyswat-greedy`.

## 13. Routing to the Expert — Zooter (Nov 2023) — [arXiv 2311.08692](https://arxiv.org/abs/2311.08692)
Distils reward-model judgements into a small router so only one LLM runs; beat the best single model on average and ranked first on 44% of 26 subsets.
- **Gist:** reward-score every LLM's responses, soften into a distribution, smooth with tag averages, train with KL.
- **Benefits:** near reward-model reranking quality at one model call.
- **How we test it:** `zooter` — RM scores from collection, tags = task/subject, soft cross-entropy; optional cost penalty λ.

## 14. Cache & Distil (Oct 2023, ACL Findings 2024) — [arXiv 2310.13561](https://arxiv.org/abs/2310.13561)
"Neural caching": a student trained online on the teacher LLM's answers takes over more traffic over time.
- **Gist:** an active-learning policy (front-loading, random, margin sampling, query-by-committee, coresets) decides which requests go to the teacher.
- **Benefits:** costs fall with no labelled data upfront; margin sampling and QBC win consistently.
- **How we test it:** `cache-distil-margin/-qbc/-front` on a replayed stream (cls tasks); live `router:cache-distil`.

## 15. EcoAssistant (Oct 2023) — [arXiv 2310.03046](https://arxiv.org/abs/2310.03046)
A ladder of assistants plus a memory of working solutions for code-driven queries; +10 points success rate over GPT-4 at under half its cost.
- **Gist:** assistant hierarchy (cheapest first, escalate on failure) + solution demonstrations retrieved for similar queries.
- **Benefits:** demonstrations make cheap models succeed more as memory grows.
- **How we test it:** `ecoassistant-hierarchy-oraclefb` (offline upper bound) and `scripts/run_ecoassistant.py` (full method live: code execution + retrieval, four configurations).

## 16. LLM Routing with Benchmark Datasets (Sep 2023) — [arXiv 2309.15789](https://arxiv.org/abs/2309.15789)
Reuse benchmark results to pick the best LLM for a new task with no labels.
- **Gist:** one kNN correctness predictor per model; scores S1 (mean prediction), S2 (corrected for out-of-distribution predictor accuracy), S3 (fall back to best-on-average when uncertain).
- **Benefits:** routers are free to build from benchmark runs you already have.
- **How we test it:** `benchmark-knn`; its report runs leave-one-task-out S1/S2/S3 model selection.

## 17. Tryage (Aug 2023) — [arXiv 2308.11601](https://arxiv.org/abs/2308.11601)
Perceptive router over a library of expert models; identified the optimal model 50.9% of the time vs 23.6% for GPT-3.5 Turbo and 10.8% for Gorilla.
- **Gist:** predict each expert's loss on the prompt; minimise predicted loss plus weighted user flags (size, recency, security, verbosity).
- **Benefits:** Pareto control over secondary goals; shines with real specialists.
- **How we test it:** `tryage` predicts gold-answer NLL per model; reports identification accuracy; add `aux.experts` (code/math models) for the expert-library setup.

## 18. LLM-Blender (Jun 2023, ACL 2023) — [arXiv 2306.02561](https://arxiv.org/abs/2306.02561)
Runs several LLMs, ranks pairwise (PairRanker), fuses the top-K (GenFuser) into an answer that beats each model.
- **Benefits:** highest quality ceiling; N calls per query — the reference point cheap routers aim at.
- **How we test it:** `llm-blender` (k best models, RM picks) + full PairRM/GenFuser point; live `router:blender`.

## 19. FrugalGPT (May 2023) — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176)
The learned LLM cascade; matched GPT-4 with up to 98% cost reduction, or +4% accuracy at equal cost.
- **Gist:** prompt adaptation, LLM approximation (caches) and a cascade whose order, scorer and thresholds are optimised under a budget.
- **Benefits:** black-box; the 98% figure comes from 10–100× API price gaps — expect smaller but real savings between open models on your GPUs.
- **How we test it:** `frugalgpt` (scorers on query + answer features, chain/threshold search) and `frugalgpt-cache` (semantic completion cache); live `router:frugalgpt` and `router:cache+<any>`.

## Side-by-side

Start with the two-tier margin rule as a free baseline, then RouteLLM-MF or EmbedLLM for pre-generation routing, and FrugalGPT or AutoMix if cascade latency is acceptable. Run the Multiple Minds report first: it shows how much headroom routing has on your data.

| Method | Pool | Training signal | Router overhead | Headline result (paper) | Best fit |
| --- | --- | --- | --- | --- | --- |
| EmbedLLM | N models | Correctness matrix | One vector op | Better accuracy and latency than prior routers | Large, changing pools |
| RouteLLM | Strong / weak | Arena preferences + golden labels | One small model | >2× cost cut at equal quality | Chat traffic, two tiers |
| MetaLLM | N models | Bandit reward (acc − cost) | Linear model | Beats single models on cost and accuracy | Classification, online feedback |
| Survey | N models | None | N calls or none | Taxonomy | Choosing the lever |
| AutoMix | Chain | Self-verification + POMDP | Small model + k verify samples | >50% compute saved | Context-grounded QA |
| OptLLM | N models | Correctness + uncertainty | Batch optimiser | 2.4–49.2% cheaper at best-LLM accuracy | Offline batch jobs |
| Two-tier | Small / large | None (logprobs) | None | Best in 25/27 setups | Strong default baseline |
| Multiple Minds | N models | Correctness | One small model | Promising, not universal | Feasibility check |
| Hybrid LLM | Small / large | Quality gap | One small model | Up to 40% fewer large calls | Quality-sensitive generation |
| Predictive Preference | N models | Pairwise preferences | Linear model | Per-prompt ranking | Preference data on hand |
| Routoo | N models | Correctness | Small LM | Mixtral quality at ⅓ lower cost | Hard budget targets |
| Fly-Swat | N models | Correctness | Meta-model + ILP | Largest-LM quality at 63% lower cost | Strict batch budgets |
| Zooter | N specialists | Reward-model scores | Small encoder | Beats best single model on average | Specialist pools |
| Cache & Distil | Student / teacher | Teacher answers, online | Tiny student | Margin and QBC win | High-volume classification |
| EcoAssistant | Chain | User feedback + code execution | Retrieval | +10 pts vs GPT-4 at <50% cost | Code and tool queries |
| Benchmark routing | N models | Benchmark results | kNN | Beats any single model | New tasks, no labels |
| Tryage | N experts | Per-model loss | Small LM | 50.9% optimal-model ID | Expert libraries |
| LLM-Blender | N models | Pairwise ranking | N calls + fuser | Beats every single model | Quality ceiling |
| FrugalGPT | Chain | Answer-reliability scorer | Scorer per stage | Up to 98% cheaper at GPT-4 quality | Large API price gaps |
