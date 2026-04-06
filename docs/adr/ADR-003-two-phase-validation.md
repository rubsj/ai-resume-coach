# ADR-003: Two-Phase Validation, Structural (Instructor) vs Semantic (Labeler)

**Date**: 2026-02-26
**Status**: Accepted
**Project**: P4 — Resume Coach

## Context

P4 validates resumes at two fundamentally different levels. Structural validation checks whether the JSON parses into a valid `Resume` Pydantic model with all 30+ required fields, correct enum values (`ProficiencyLevel`, `ExperienceLevel`), and passing field validators (ISO dates, GPA 0.0 to 4.0, phone >= 10 digits). Semantic validation checks whether the content is reasonable: does a resume claim 15 years of experience for a Junior-level role, does it list skills that never appeared in the job description, does the writing feel like awkward AI-generated prose.

These are different failure modes with different detection costs. Structural failures are binary (Pydantic accepts or rejects). Semantic failures require comparing resume content against the paired job description using domain-specific rules (Jaccard similarity, experience year thresholds, seniority level mapping).

Mixing both into one layer creates three problems. First, cost: putting semantic rules into Instructor retry prompts means every labeling check costs $0.002/pair in API calls for comparisons that Python can compute in 250ms total for all 250 pairs. Second, testability: if the labeler requires an LLM to run, `test_labeler.py` needs API mocking or real API calls, and P4 has 532 tests that should all run without API keys. Third, coupling: the generator would need to know about business rules (experience thresholds, seniority mappings) to inject them as retry prompts, and the labeler would need to parse raw JSON. Each module would do two jobs instead of one.

## Decision

I implemented two-phase validation with strict separation.

Phase 1 (structural) runs at generation time in `generator.py` via `instructor.from_openai(client, mode=Mode.JSON)` with `max_retries=5`. It validates against 35 Pydantic models with ~30 validation points per `Resume`. On failure, Instructor injects the exact `ValidationError` back as a correction prompt and retries. Output is guaranteed valid `GeneratedResume` and `GeneratedJob` objects. Downstream stages never see malformed data. Cost is ~1 to 2 extra API calls on the ~15% of resumes that fail first-pass validation.

Phase 2 (semantic) runs post-generation in `labeler.py` via `label_pair(resume: Resume, job: JobDescription, pair_id: str, normalizer: SkillNormalizer)`. It computes 5 boolean failure flags (`experience_mismatch`, `seniority_mismatch`, `missing_core_skills`, `has_hallucinations`, `has_awkward_language`) plus supporting numeric fields (`skills_overlap` as Jaccard 0.0 to 1.0, `experience_years_resume`, `seniority_level_resume` 0 to 4, `missing_skills: list[str]`). `FailureLabels` has 18 fields total. This phase is pure Python: deterministic, ~1ms per pair, zero LLM calls, zero cost.

An optional Phase 3 runs in `judge.py` when deeper quality assessment is needed. It uses GPT-4o via Instructor for structured output: `has_hallucinations`, `has_awkward_language`, `overall_quality_score` (0 to 1), `fit_assessment`, `recommendations`, `red_flags`. Cost is ~$0.002/pair x 250 = ~$0.50, skippable via `--skip-judge` flag. Phase 3 provides a second opinion on Phase 2's rule-based labels, used for agreement analysis (`judge_vs_rules_agreement.png`).

## Alternatives Considered

**Single-phase in Instructor (validate everything at generation time)** - This would embed semantic rules like "experience must be >= 3 years for senior role" into the LLM retry loop. Instructor's retry mechanism works well for binary constraints (field present/absent, enum valid/invalid) but struggles with judgment calls like "is this experience mismatch bad enough to flag?" Mixing both types means the model could fix a semantic issue by introducing a structural one. It also means labeling costs $0.002/pair in API calls for checks that Python can do in 1ms.

**Single-phase in labeler (validate everything post-generation)** - The labeler would handle both JSON parsing and content analysis. But the labeler receives deserialized Pydantic objects; it never sees raw JSON. To also handle structural validation, it would need to accept raw strings, wrap `model_validate_json()` with fallback logic, and return partially-parsed objects. The labeler's `label_pair()` signature takes typed `Resume` and `JobDescription` objects precisely because Phase 1 guarantees they're valid.

**LLM for all validation** - Maximum flexibility, but at 10 to 100x the cost of rule-based checks, with non-deterministic outputs that change between runs, and ~1s latency per pair instead of ~1ms. P4's failure mode analysis requires reproducible numbers. Running the labeler twice should produce identical results. An LLM-only approach makes that impossible.

## Quantified Validation

Phase 2 labeling processes all 250 pairs in ~250ms at $0.00. Phase 3 judge is ~$0.50 and skippable via `--skip-judge`. All 532 tests pass without API keys, and `test_labeler.py` covers all 5 failure modes with pure Python fixtures, no mocking, no flakiness. The corrector (8/8 records fixed, 100% rate) operates on Phase 2 labels and never needs to re-validate structure because Phase 1 already guaranteed it.

## Consequences

Because structural validation runs at generation time via Instructor, every downstream module (labeler, judge, corrector, analyzer, multi_hop) receives guaranteed-valid typed objects. No defensive parsing, no `try/except` around `model_validate()`, no "what if this field is None?" guards in 5 separate modules. The phase separation also enabled the judge-vs-labeler agreement analysis: comparing Phase 2's rule-based labels against Phase 3's GPT-4o assessments is only meaningful because the two phases produce genuinely independent evaluations.

Two separate pipeline stages means two invocation points: `run_generation.py` (Phase 1) and `run_labeling.py` (Phase 2), orchestrated by `pipeline.py`. Adding a new validation concern requires deciding which phase it belongs in. Edge cases that are structurally valid but semantically absurd (a date range `"2020-01"` to `"2019-06"` that passes ISO validation but is logically backwards) fall between phases; I handle these with explicit checks in the labeler, but they're easy to miss. The `--skip-judge` flag means some pipeline runs have Phase 3 data and some don't, so downstream code must handle the missing-judge-data case.

P1 mixed structural and semantic validation in a single `evaluator.py`, which worked at small scale (30 records, 6 failure modes) but made testing expensive since every evaluator test needed either a real LLM call or complex mocking. P4's two-phase split is the direct response to that problem at 8x the scale. (This maps to Java's separation of `@Valid` on a `@RequestBody` DTO from `@Service` business rules: structural validation fails fast at the boundary, domain logic runs on guaranteed-clean data downstream.)
