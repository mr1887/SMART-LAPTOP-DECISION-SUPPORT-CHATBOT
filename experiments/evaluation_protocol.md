# Experimental Protocol

## Research focus
Evaluate the feasibility of the hybrid LLM + Constraint Programming architecture for Vietnamese laptop decision support.

## Research questions
- **RQ1 – Technical feasibility:** Can the architecture reliably transform natural-language requirements into structured constraints and produce feasible decisions?
- **RQ2 – Decision reliability:** Compared with LLM-only, does the hybrid architecture improve constraint compliance and product validity?
- **RQ3 – Operational feasibility:** What trade-offs arise in latency, LLM usage, token usage, and operating cost?

## Benchmark design
- 60 query-turns.
- 52 independent single-turn cases.
- 8 multi-turn turns across 3 conversations.
- 8 intentionally infeasible cases.
- Ground truth is fixed before running the systems.
- Multi-turn cases are used mainly for RQ1 and have `compare_with_baseline=0`.

## Ground-truth fields
Each benchmark row fixes:
- required use-case tags;
- hard and soft constraints;
- preferences;
- expected feasibility;
- expected context action;
- expected candidate count under the processed catalog.

## Main protocol
1. Freeze application code and model versions.
2. Run the pilot subset first.
3. Inspect only experiment plumbing: crashes, missing logs, malformed outputs, or evaluator errors.
4. Do not tune recommendation logic after seeing pilot quality unless there is a genuine implementation bug.
5. Run the full benchmark using the same LLM model/version and temperature.
6. Preserve all failures and difficult cases.
7. Analyze results with the independent evaluator.
8. Report multi-turn results separately from the main LLM-only comparison.

## RQ1 metrics
- End-to-end success rate.
- Exact tag extraction rate.
- Exact constraint extraction rate.
- Full RequirementSet exact rate.
- Feasibility/infeasibility detection accuracy.
- Candidate-count consistency.
- ADD / UPDATE / REPLACE accuracy on multi-turn cases.

## RQ2 metrics
Main compliance rates are reported on ground-truth feasible cases.
- Hard Constraint Satisfaction Rate.
- Hard Constraint Violation Rate.
- Soft Constraint Violation Rate.
- Product Validity Rate.
- Hallucination Rate.

The evaluator independently maps the recommended product back to the catalog and re-checks constraints instead of trusting the solver's own status flag.

## RQ3 metrics
- Average, median, and p95 latency.
- Average API attempts.
- Input, output, thinking, and total tokens from Gemini `usage_metadata`.
- Average estimated API cost per query.
- Total estimated API cost for the benchmark.
- Cost per completed recommendation.
- Cost per catalog-valid recommendation.

### API cost calculation
For each Gemini response:

```
estimated_cost_usd =
    input_tokens * input_price_per_1M / 1_000_000
  + (output_tokens + thinking_tokens) * output_price_per_1M / 1_000_000
```

The experiment snapshot uses paid-tier standard prices as of **2026-10-20**:
- `gemini-3.8-flash`: USD 0.75 / 1M input tokens, USD 3.75 / 1M output tokens.
- `gemini-3.6-flash`: USD 0.75 / 1M input tokens, USD 3.75 / 1M output tokens.
- `gemini-3.5-flash-lite`: USD 0.30 / 1M input tokens, USD 2.50 / 1M output tokens.

Gemini 3.8/3.6 Flash rates are introductory prices valid through 2026-12-31,
so the pricing date must be reported with the research results. The built-in
rates live in `backend/app/evaluation/cost.py` and can be overridden through
environment variables if the experiment is rerun under a different price schedule.

## Supplemental RQ3 cost run after the main experiment

If the main benchmark was already completed before token/cost logging was added,
do **not** overwrite the frozen `results.jsonl`. Run a separate operational-cost
experiment on the same cases used for the LLM-only vs Hybrid comparison.

Recommended command:

```bash
python experiments/run_experiment.py \
  --systems both \
  --comparison-only \
  --output experiments/results/cost_rerun.jsonl \
  --baseline-delay 6
```

This selects only rows with `compare_with_baseline=1` (52 cases in the current
benchmark), so both systems are measured on the same fixed queries. It excludes
the 8 multi-turn turns that are mainly used for RQ1.

If rate limits make one-shot execution inconvenient, run the same 52 cases in
batches without touching the frozen main results:

```bash
python experiments/run_experiment.py --systems both --comparison-only --output experiments/results/cost_rerun.jsonl --batch-number 1 --batch-size 10 --baseline-delay 6
python experiments/run_experiment.py --systems both --comparison-only --output experiments/results/cost_rerun.jsonl --batch-number 2 --batch-size 10 --baseline-delay 6
# Continue until the final batch.
```

Then analyze the supplemental run separately:

```bash
python experiments/analyze_results.py \
  --results experiments/results/cost_rerun.jsonl \
  --evaluated experiments/results/cost_rerun_evaluated.csv \
  --summary experiments/results/cost_rerun_summary.json
```

For the report, use the original frozen experiment for the primary RQ1/RQ2
results. Use `cost_rerun_summary.json` only as the supplementary evidence for
RQ3 token usage, API cost, and operational efficiency. This separation avoids
retroactively changing the already-completed main experiment.

## LightGBM
LightGBM is a learning-based global relevance estimator inside the proposed architecture, not a separate research question.

Its role is:
`Global relevance (LightGBM) + query-specific matching -> utility -> CP-SAT decision`.

Optional only if time remains: ablation of the full Hybrid system versus Hybrid without LightGBM.

## Catalog-grounded LLM baseline

A third system is included to make the RQ2 comparison fairer:

- **LLM-only:** no study catalog, no symbolic optimizer.
- **Catalog-grounded LLM:** receives the complete compact study catalog and must
  choose exactly one catalog product, but uses no LightGBM, retrieval, or OR-Tools.
- **Hybrid:** catalog-grounded pipeline with structured NLU, scoring/retrieval,
  and CP-SAT decision logic.

This separates two effects:

1. `LLM-only -> Catalog-grounded LLM`: benefit of grounding the decision to the
   study product space.
2. `Catalog-grounded LLM -> Hybrid`: additional benefit of structured symbolic
   decision-making after both systems have access to the same product catalog.

The grounded prompt contains every catalog row but only decision-relevant fields
(name/id, price, CPU/GPU, weight, battery, available RAM/storage fields, screen,
and use-case tags). Internal Hybrid scores such as LightGBM relevance are not
provided to this baseline.

### Recommended run

Do not rerun the already frozen LLM-only/Hybrid benchmark just to add this
baseline. Run the 52 comparison cases into a separate file:

```bash
python experiments/run_experiment.py \
  --systems catalog_grounded_llm \
  --comparison-only \
  --output experiments/results/catalog_grounded_llm.jsonl \
  --baseline-delay 6
```

For safer incremental execution:

```bash
python experiments/run_experiment.py --systems catalog_grounded_llm --comparison-only --output experiments/results/catalog_grounded_llm.jsonl --batch-number 1 --batch-size 10 --baseline-delay 6
python experiments/run_experiment.py --systems catalog_grounded_llm --comparison-only --output experiments/results/catalog_grounded_llm.jsonl --batch-number 2 --batch-size 10 --baseline-delay 6
# Continue through batch 6 (the last batch contains 2 cases).
```

Analyze it independently:

```bash
python experiments/analyze_results.py \
  --results experiments/results/catalog_grounded_llm.jsonl \
  --evaluated experiments/results/catalog_grounded_llm_evaluated.csv \
  --summary experiments/results/catalog_grounded_llm_summary.json
```

Use the same RQ2/RQ3 metrics as Hybrid. Because the selected product is expected
to come from the study catalog, hard/soft constraint compliance is independently
rechecked against catalog data rather than trusted from the LLM's self-claimed
specifications.

## Baseline interpretation
The direct LLM-only baseline remains useful for measuring the effect of the complete grounded architecture. However, causal claims about the optimizer should rely primarily on the Catalog-grounded LLM vs Hybrid comparison, because both systems operate over the same study catalog. Even then, the Hybrid also includes structured NLU/scoring in addition to CP-SAT, so avoid attributing every difference exclusively to the optimizer.

## Interpretation rule
Conclusions must be limited to this catalog, these Vietnamese queries, and the evaluated model configuration. Do not claim universal superiority.


## API failure handling
- LLM-only API failures are recorded as `API_FAILED`, not as hallucinations.
- Responses returned by the API but not parseable into the required JSON/product are recorded as `PARSE_FAILED`.
- Product validity and hallucination rates are computed only over completed recommendations; API/parse failure rates are reported separately.
- The runner records the successful Gemini model and the number of API attempts for reproducibility.
- Hybrid runs record `nlu_source` as `gemini_plus_regex` or `regex_fallback` so RQ1 does not silently attribute fallback behavior to Gemini.
- Use a delay between baseline cases when necessary to avoid burst rate-limit artifacts; the recommended full-run setting is `--baseline-delay 6` seconds.


## Catalog validity vs hallucination
- A completed recommendation that matches the evaluation catalog is labeled `CATALOG_VALID`.
- A completed recommendation that does not match the evaluation catalog is labeled `OUT_OF_CATALOG`.
- `OUT_OF_CATALOG` must **not** be automatically interpreted as hallucination, because a model may recommend a real product that is simply absent from the study catalog.
- `VERIFIED_HALLUCINATION` requires independent external verification that the product/model is fabricated or materially nonexistent. The automatic evaluator therefore reports verified hallucination as unavailable unless such verification is supplied.
- RQ2 should primarily report catalog validity, out-of-catalog recommendation rate, hard-constraint satisfaction, and soft-constraint violations. Any hallucination claim must be separately verified and described as such.


## RQ2 verification policy (final)
To avoid conflating catalog coverage with decision correctness:

- `OUT_OF_CATALOG` is **not** counted as a hard-constraint violation and is **not** counted as hallucination.
- Independent catalog verification is computed only when the recommended product can be mapped to the study catalog.
- For LLM-only outputs that are outside the catalog, explicit constraints may be assessed from the model's own `claimed_specs`, but this is reported only as **claim-based compliance**. It is not factual verification.
- Missing fields (for example battery duration when the baseline schema does not provide it) are labeled `UNVERIFIABLE`, not false.
- Semantic usage tags such as office/programming/gaming/graphics are not inferred from LLM self-claims for independent verification.
- True hallucination requires independent external verification of product existence or materially false specifications.

The primary RQ2 evidence should therefore separate:
1. catalog validity / out-of-catalog rate;
2. catalog-verified satisfaction of hard requirements on catalog-mapped recommendations;
3. explicit hard/soft constraint compliance with the evidence source clearly labeled (`independent_catalog` vs `llm_self_claimed_specs`);
4. independently verified hallucination, if external verification is performed.

## RQ3 operational policy (final)
- Latency must report mean, median, and p95 because API/model fallback can create large outliers.
- API effort must use actual attempts consistently: `api_attempts` for LLM-only and `nlu_api_attempts` for Hybrid.
- A logical pipeline call count must not be compared directly against baseline retry attempts.
- Token/cost conclusions are valid only for runs where Gemini `usage_metadata` is actually logged.
- Cost values are estimates under the documented paid-tier price snapshot; they are not billing invoices.
- Thinking tokens are charged with output tokens and are included in estimated output cost.


## Strict catalog identity matching
- Exact normalized product-name matches are accepted directly.
- Substring/fuzzy matches are accepted only when model-identity tokens are compatible.
- Concrete model identifiers asserted by the LLM (for example Apple M-generation, ThinkPad Gen number, ASUS/Lenovo SKU-like alphanumeric codes) must also appear in the catalog product name.
- This intentionally prefers false negatives over false positives for catalog-validity measurement.
- Examples that must **not** be auto-matched: M3 Pro -> M5 Pro, Gen 5 -> Gen 7, 16APH8 -> 16AHP9, or a specific UX3402ZA model -> a generic Zenbook family entry.


## Full external verification of LLM-only outputs
The repository now includes `experiments/external_verification_llm_only.csv`, covering all 52 completed LLM-only benchmark outputs.

Labels:
- `VERIFIED_REAL`: the product/model is supported by an external source.
- `EXACT_MODEL_NOT_CONFIRMED`: the product family exists, but the exact model/SKU could not be confirmed from the checked sources. This is unresolved and must **not** be counted as hallucination.
- `NON_PRODUCT_RESPONSE`: the LLM did not recommend a product (for example, it abstained on an infeasible request).
- `VERIFIED_HALLUCINATION`: reserved for cases where external evidence supports that the named product/model is fabricated or nonexistent.
- `MATERIAL_SPEC_ERROR`: the product exists, but one or more important self-claimed specifications (e.g. CPU/GPU) conflict with external evidence.
- `PARTIAL_CHECK_ONLY`, `MODEL_SUFFIX_OMITTED`, and `CONFIG_AMBIGUOUS`: verification was incomplete or configuration-dependent, so these must be interpreted conservatively.

RQ2 must report product-existence hallucination separately from specification errors. Search absence alone is not enough to label a hallucination.
