# ADR-002: Custom Skill Normalizer over Third-Party Library

**Date**: 2026-02-26
**Status**: Accepted
**Project**: P4 — Resume Coach

## Context

Jaccard similarity between resume skills and job requirements is P4's core metric for fit scoring. The Jaccard gradient across fit levels is the central hypothesis, the `missing_core_skills` failure label uses set difference, and the A/B template comparison measures failure rates derived from these labels.

Without normalization, the metric collapses. A resume listing `"Python 3.11"` against a job requiring `"Python"` scores Jaccard = 0.0, a false negative on what should be a perfect match. `"ML"` vs `"Machine Learning"`, `"React.js"` vs `"React"`, `"Node.js"` vs `"Node"` all produce zero overlap. Since GPT-4o-mini generates skills with natural variation (version numbers, abbreviations, framework suffixes), unnormalized Jaccard would measure formatting noise, not actual skill fit.

The 250 resumes average ~8 skills each. Across 250 resume-job pairs, that's ~4,000 skill comparisons where normalization matters. The normalizer must be deterministic (same input always produces same output), fast (runs inside the labeler at ~1ms per pair), and free (no API calls; the labeler is pure Python by design, per ADR-003).

## Decision

I built a custom `SkillNormalizer` class with a five-stage pipeline, applied to every skill string before Jaccard computation:

1. Lowercase + strip: `"Python"` becomes `"python"`, `"  React "` becomes `"react"`
2. Parenthetical removal: `"JavaScript (ES6+)"` becomes `"javascript"` via regex `r"\s*\([^)]*\)"`
3. Version stripping: `"python 3.10"` becomes `"python"`, `"react 18"` becomes `"react"` via regex `r"\s*\d+(\.\d+)*\s*$"`
4. Suffix removal: strip 7 noise words (`developer`, `engineer`, `programming`, `development`, `framework`, `language`, `library`) from the end of skill strings
5. Alias resolution: 28-entry canonical mapping (`{"ml": "machine learning", "js": "javascript", "k8s": "kubernetes", "react.js": "react", "node.js": "node", "aws": "amazon web services", ...}`)

The normalizer is instantiated once at process startup (API, Streamlit, or pipeline) and passed into `label_pair(resume, job, pair_id, normalizer)`. The alias dict and suffix list are plain Python data structures, no config files, no model weights.

## Alternatives Considered

**skillNer (spaCy-based NER)** - Industry-standard skill extraction that recognizes skill entities in free text. Requires downloading a spaCy language model (~50MB), which is larger than P4's entire dataset. More importantly, skillNer solves a different problem: extracting skills from unstructured text. P4's skills are already structured (they come from Pydantic `Skill.name` fields). The bottleneck is normalization of known skill strings, not extraction from prose. Using NER for this is like using OCR to read a database column.

**rapidfuzz (fuzzy string matching)** - Handles typos elegantly with Levenshtein distance. But fuzzy matching is dangerously imprecise for skills: `"Java"` and `"JavaScript"` score ~82% similarity by Levenshtein, despite being completely different languages. Any threshold that catches `"JS"` to `"JavaScript"` also catches `"Java"` to `"JavaScript"`. The false positive rate would corrupt the Jaccard gradient, the exact metric P4 exists to measure.

**OpenAI embeddings cosine similarity** - Semantic embeddings would catch every variant, including ones the alias dict misses. But normalizing 250 resumes x ~8 skills means ~2,000 API calls per pipeline run at $0.02 per 1K tokens. The labeler runs inside a tight loop and must be free and deterministic (ADR-003). Embedding-based normalization would make the labeler non-deterministic (model updates change results) and add ~$0.04 per run, which is small but breaks the design principle that labeling is pure Python with zero external dependencies.

## Quantified Validation

The Jaccard gradient across fit levels is near-monotonic: excellent=0.669, good=0.607, partial=0.620, poor=0.212, mismatch=0.005. The partial/good inversion (0.620 vs 0.607) is likely due to template variation at mid-range fit levels, but the endpoints confirm the trend: excellent-fit resumes share 67% of required skills while mismatch resumes share effectively none. Without normalization, this gradient would be noise. The `missing_core_skills` failure label fires at 50.8% across 250 pairs and produces zero false positives from formatting noise because both sides are canonicalized before comparison. The normalizer processes all 250 pairs in ~250ms at zero cost, and `test_normalizer.py` is fully deterministic with exact input-to-output assertions. The A/B template test found statistically significant differences between templates (chi-squared=32.74, df=4, p=1.35e-06), which depends on accurate failure labels, which depend on accurate Jaccard scores, which depend on the normalizer.

## Consequences

The normalizer is the foundation of P4's analytical validity. Every Jaccard score, every failure label, and every chart in the analysis flows through it. `test_labeler.py` and `test_normalizer.py` are fully deterministic with no mocking and no flakiness.

The 28-entry alias dict requires manual maintenance. Adding a new domain (biotech skills, for example) means updating `SKILL_ALIASES` by hand. I mitigated this by covering the top ~50 tech skills that appear in GPT-4o-mini's generation vocabulary, since the LLM rarely invents abbreviations outside that set. Typos in generated skills (like `"Pythn"`) are not caught, but GPT-4o-mini's spelling is reliable enough that this hasn't produced a false negative in 250 resumes. The parenthetical removal regex `r"\s*\([^)]*\)"` strips all parenthetical content, which could remove meaningful qualifiers like `"SQL (advanced)"`. This is acceptable because `proficiency_level` is a separate field on the `Skill` model, so the parenthetical is redundant. P1 had no skill normalization since its repair records used free-text descriptions, not structured skill lists. P4's Day 1.5 data quality audit revealed near-zero Jaccard overlap from un-normalized skills, making the normalizer a direct response to that failure. (This is a stateless service in Spring terms: instantiated once, injected into dependent services, no mutable state, with the five-stage pipeline mapping to a `Stream` chain.)
