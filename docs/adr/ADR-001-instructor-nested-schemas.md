# ADR-001: Instructor with max_retries=5 for Nested Schema Generation

**Date**: 2026-02-26
**Status**: Accepted
**Project**: P4 — Resume Coach

## Context

P4 generates 250 resumes as deeply nested Pydantic models. A single `Resume` has 4 levels of nesting: `ContactInfo`, `list[Education]` (each with optional `gpa` constrained to 0.0-4.0 and ISO-format `graduation_date`), `list[Experience]` (each with `responsibilities: list[str]` requiring min 1 item and dates validated in sequence), and `list[Skill]` (each with a `ProficiencyLevel` enum and `years` constrained to 0-30). Across the full schema there are roughly 30 individual validation points where raw GPT-4o-mini output can fail Pydantic.

Raw OpenAI completions fail validation 15-30% of the time on schemas this complex. Common breakages: `null` instead of `[]` for lists, `"advanced"` instead of the `ProficiencyLevel.ADVANCED` enum literal, `"2020"` instead of ISO `"2020-01"` for dates, missing required fields in nested objects. A naive retry-from-scratch strategy wastes API calls because the model gets no feedback on what went wrong.

P1 had ~7 validation points per record and could get away with simpler prompting. P4's 4x increase in schema complexity made an intelligent retry mechanism a hard requirement.

## Decision

I used **Instructor** (`instructor.from_openai(client, mode=instructor.Mode.JSON)`) with `max_retries=5` for all structured LLM generation.

The core value is automatic error injection: when Pydantic validation fails, Instructor extracts the `ValidationError` message and injects it back into the conversation as a correction prompt. The model sees exactly which field failed and why (`"1 validation error for Resume: skills.0.years, Value error, years must be 0-30"`) rather than just retrying blind. This turns a blind retry into a targeted fix.

The entire retry loop is `max_retries=5` on the `create()` call. No `try/except`, no backoff, no prompt rewriting. The generator functions (`generate_job`, `generate_resume`) stay under 50 lines instead of becoming 100-line retry state machines.

`Mode.JSON` forces the API to return valid JSON at the protocol level, eliminating failures where the model prefixes output with prose like `"Here is the JSON:"`. On the schema side, fields that LLMs rarely produce correctly (`linkedin`, `portfolio`, `coursework`) are marked `Optional[str] = None` so their absence doesn't trigger retries. `responsibilities: list[str]` has no `max_length` because constraining list length caused more retries than it prevented bad data.

## Alternatives Considered

**Raw OpenAI + manual json.loads()** - The caller parses the raw string, validates with `Resume.model_validate(data)`, and when validation fails (15-30% of the time on 30-field schemas), must catch `JSONDecodeError` and `ValidationError` separately, format the error into a correction prompt, append it to conversation history, and re-call the API. In P1 this was viable because `RepairRecord` had 7 fields and the retry logic was a 10-line `for` loop. With P4's 35 Pydantic models and 30 validation points per resume, the manual retry logic would be the largest source of complexity in the generator, easily 100+ lines of error handling per generation function, repeated across `generator.py`, `judge.py`, and `corrector.py`.

**LangChain PydanticOutputParser** - LangChain's `OutputFixingParser` re-sends the entire malformed output with a generic instruction ("Fix the following output to match the schema") but does not include the specific `ValidationError`. For a schema with 30 validation points, the model often fixes one obvious issue while introducing a new one. Instructor's approach is different: it extracts the exact `ValidationError` message (`"skills.0.years: Value error, must be 0-30, got -1"`) and injects it as a user message, giving the model a field-level correction target. That's the difference between "your JSON is wrong" and "field X at path Y failed constraint Z with value W."

**Pydantic model_validate_json() with hand-rolled retry** - Pure stdlib approach with no third-party dependency beyond Pydantic. But the entire value of Instructor is the correction prompt construction: parsing `ValidationError.errors()` into dicts, formatting each into natural language the model can act on, appending to the conversation, and tracking multi-turn history. Building this manually means writing `format_validation_errors()`, a conversation history manager, and a retry controller, roughly 80-100 lines of infrastructure code identical across three pipeline stages. Instructor replaces all of that with one parameter.

## Quantified Validation

100% validation rate across 250 resumes, each with 30+ validation points across 35 Pydantic model types, with zero manual intervention. Roughly 15% of resumes needed 1-2 retries to pass, but none exhausted all 5 attempts because Instructor's error injection gives the model a targeted correction path. The same pattern carried across three pipeline stages: `generator.py` for resume/job creation, `judge.py` for GPT-4o structured evaluation, and `corrector.py` for the correction loop. Worst-case 5x API cost per record on persistent failures added ~$0.02 across the full 250-resume run, negligible against the ~$0.50 total generation cost.

## Consequences

Generator functions stay under 50 lines each: no try/except blocks, no backoff logic, no prompt rewriting. The Instructor pattern reused cleanly across all three pipeline stages without modification.

Instructor pins to a specific OpenAI client version. Major `openai` SDK bumps (v0 to v1 was painful for the community) require waiting for an Instructor update. I mitigated this by pinning both in `pyproject.toml`. `Mode.JSON` doesn't support streaming, which is acceptable since P4 generation is batch, not interactive. P1 used raw `model_validate_json()` with manual retry for its simpler 7-field `RepairRecord` schema, which revealed that retry prompt engineering, not parsing, was the real bottleneck. Instructor was the direct response to that lesson. (This is analogous to combining Jackson's `@JsonCreator` with Spring Retry's `@Retryable`, except Instructor feeds the specific validation error back to the LLM as correction context rather than retrying blind.)
