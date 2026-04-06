# ADR-004: FastAPI over Flask for REST API

**Date**: 2026-02-26
**Status**: Accepted
**Project**: P4 — Resume Coach

## Context

P4 needs a REST API exposing 9 endpoints across 4 domains: resume review (live labeling + optional GPT-4o judge), pipeline analysis (failure rates, A/B template comparison), vector search (ChromaDB semantic similarity), and feedback (human ratings). The API serves as the backend for the Streamlit demo, with interactive Swagger documentation at `/docs`.

P4 already has 14 Pydantic request/response schemas defined in `schemas.py`: `ReviewRequest`, `ReviewResponse`, `FailureRateResponse`, `TemplateComparisonResponse`, `MultiHopRequest`, `MultiHopResponse`, `SimilarCandidatesResponse`, `FeedbackRequest`, `FeedbackResponse`, `JobListResponse`, `PairDetailResponse`, plus supporting models like `TemplateStats`, `MultiHopQuestion`, `SimilarCandidate`, `JobSummary`. These schemas define the API contract. The framework must consume them directly; writing adapter DTOs or serialization boilerplate for 14 schemas would double the code without adding value.

P4's data lives in JSONL files on disk, loaded into memory at startup by `DataStore`. There is no database, no ORM, no migrations. The framework must work with a stateless, file-backed architecture.

## Decision

I used **FastAPI** for the Resume Coach REST API.

The 14 Pydantic schemas plug in with zero adapter code. `ReviewRequest` is both the Pydantic validation model and the FastAPI request body parameter. `response_model=ReviewResponse` on the decorator is both the serialization format and the OpenAPI schema. In Flask, each of these 14 schemas would need a manual `request.get_json()` into `model_validate()` round-trip on input and a `model_dump()` call on output, roughly 28 lines of boilerplate across 9 endpoints. FastAPI eliminates all of it.

Swagger UI at `/docs` renders all endpoints interactively the moment the server starts, with zero configuration. With Flask, achieving the same requires installing `flask-restx` or `flask-smorest`, writing `@api.doc()` decorators, and maintaining schema descriptions separately from the Pydantic models.

`Annotated[int, Query(ge=1, le=50)]` on query parameters (like `page_size` on `/jobs`) expresses validation constraints inline with the type. FastAPI auto-generates the constraint documentation in Swagger and returns structured 422 errors on violation. The Flask equivalent is three lines of manual validation per parameter: `page_size = request.args.get('page_size', 20); page_size = int(page_size); if page_size < 1 or page_size > 50: abort(422)`.

## Alternatives Considered

**Flask** - Simpler, lighter, and the most familiar Python web framework. But Flask's design assumes you build request parsing yourself. Without native Pydantic support, every endpoint needs `request.get_json()`, `try/except` around `model_validate()`, manual `model_dump()` for the response, and manual error formatting for validation failures. For 9 endpoints with 14 schemas, this adds ~120 lines of boilerplate that FastAPI handles automatically. Flask also lacks built-in OpenAPI generation, and `flask-restx` requires duplicate schema definitions in marshmallow format, defeating the purpose of having Pydantic models in `schemas.py`.

**Django REST Framework** - The most full-featured Python API framework, with authentication, ORM integration, admin panels, and browsable API. But P4 has no database. Data comes from JSONL files loaded into `DataStore` at startup. Django's ORM, migration system, and model serializers would be unused dead weight. The `django.contrib.*` dependencies alone are larger than P4's entire codebase.

**Litestar** - The fastest Python web framework by benchmark, with auto-generated OpenAPI and native support for both dataclasses and Pydantic. A legitimate contender on technical merits. But Litestar's design favors a dataclass-first approach: its dependency injection, DTOs, and middleware are built around `@dataclass` and `msgspec.Struct` types, with Pydantic support added as a plugin. P4's 14 Pydantic v2 schemas serve triple duty (LLM structured output via Instructor, API contracts, and pipeline validation). Adapting these to Litestar's DTO layer would require either wrapping each Pydantic model in a Litestar-specific DTO class or configuring the Pydantic plugin per endpoint. That friction doesn't exist in FastAPI where Pydantic is the native type system. Litestar's performance advantage (microseconds per request) is also irrelevant when endpoints call GPT-4o at 1s+ latency.

## Quantified Validation

The 346-line `api.py` serves 9 endpoints using all 14 Pydantic schemas from `schemas.py` with zero adapter code. Adding the `/feedback` endpoint (POST with `FeedbackRequest` body, returns `FeedbackResponse`) took 15 lines including file I/O because the framework handled parsing, validation, serialization, and documentation automatically. Structured 422 errors on invalid input come free with no custom error handling code. All 26 endpoint tests run without starting a server process via FastAPI's `TestClient` (wraps httpx), contributing to the 532-test suite that runs in ~2 seconds.

## Consequences

Every Pydantic schema is both the FastAPI parameter type and the OpenAPI documentation source, so the same `Resume`, `FailureLabels`, and `JudgeResult` models created for batch processing became API contracts with zero refactoring. P1 and P2 were batch pipelines with no API layer; P4 is the first project serving results over HTTP, so this decision had to carry all 14 schemas without duplication.

FastAPI's dependency injection system (`Depends()`) has a learning curve for shared state patterns beyond module-level singletons. P4 uses three module-level singletons (`_store: DataStore`, `_normalizer: SkillNormalizer`, `_collection: chromadb.Collection`), which is simple but won't scale to multi-tenant or request-scoped state. `TestClient` requires `httpx` as a dev dependency. FastAPI's startup is heavier than Flask (~200ms vs ~50ms) due to OpenAPI schema generation, which is irrelevant for a long-lived server but noticeable in test suites that create the app per test class. (This is what Spring Boot would be if Jackson, Bean Validation, and springdoc-openapi were a single annotation on the route decorator.)
