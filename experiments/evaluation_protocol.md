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
- Average and median latency.
- Average LLM calls.
- Input/output tokens when API metadata is available.
- Estimated API cost when pricing/token metadata is available.

## LightGBM
LightGBM is a learning-based global relevance estimator inside the proposed architecture, not a separate research question.

Its role is:
`Global relevance (LightGBM) + query-specific matching -> utility -> CP-SAT decision`.

Optional only if time remains: ablation of the full Hybrid system versus Hybrid without LightGBM.

## Baseline interpretation
The existing baseline is direct LLM-only recommendation. Therefore any product-validity advantage of the Hybrid system reflects the complete grounded architecture, not CP-SAT alone. State this explicitly as a limitation; do not attribute every difference causally to the optimizer.

## Interpretation rule
Conclusions must be limited to this catalog, these Vietnamese queries, and the evaluated model configuration. Do not claim universal superiority.
