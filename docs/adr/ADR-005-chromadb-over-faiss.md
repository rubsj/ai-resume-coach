# ADR-005: ChromaDB over FAISS for Vector Search

**Date**: 2026-02-26
**Status**: Accepted
**Project**: P4 — Resume Coach

## Context

P4 requires semantic search over 250 resumes: given a free-text job description or skill query, return the top-k most similar candidates ranked by embedding cosine similarity. This powers the `/search/similar-candidates` API endpoint and the "Search Similar" page in the Streamlit demo.

Three requirements made the choice non-trivial. First, persistence: the FastAPI process restarts frequently during development (`uvicorn --reload`), and encoding 250 resumes with `all-MiniLM-L6-v2` takes ~2 seconds on the M2 MacBook Air. Rebuilding the index on every restart means a 2-second delay on the first request, which feels broken in a demo. Second, metadata filtering: the API supports filtering search results by `fit_level` ("excellent", "good", "partial", "poor", "mismatch"), and without native filtering every query would return all 250 results for Python-side post-filtering. Third, live API integration: the vector store must be queryable from a long-lived FastAPI process with `count()` for the health endpoint, `query()` for search, and `get_or_create_collection()` for startup.

In P2 (RAG Evaluation), I used FAISS for batch document retrieval. FAISS is an in-memory index: fast but ephemeral. It met P2's needs (build index, run evaluation, exit). P4's live-serving requirements are fundamentally different.

## Decision

I used **ChromaDB** (`PersistentClient` at `data/chromadb/`) with `all-MiniLM-L6-v2` embeddings (384 dimensions, cosine space).

The lifecycle is two-phase. `build_resume_index()` runs once after the pipeline completes: loads the SentenceTransformer model, encodes all 250 resumes in batch, stores embeddings + metadata in ChromaDB, then `del model; gc.collect()` to free ~90MB on the 8GB M2. `get_collection()` runs at every API/Streamlit startup: opens the persisted index with zero model loading for instant startup.

A `_SentenceTransformerEF` class wrapping `SentenceTransformer.encode()` is loaded once as a module-level singleton at first query time and stays in memory. This avoids the ~2s model reload on every search request while keeping startup instant (model loads lazily on first search, not at import time).

ChromaDB 1.5 has a known conflict: passing a custom `EmbeddingFunction` to `get_collection()` on a collection built without one raises `ValueError`. The workaround: I manage `_ef` as my own singleton, call `_ef([query_text])` to get the embedding, and pass it to `collection.query(query_embeddings=...)` directly, bypassing ChromaDB's internal EF dispatch entirely.

Each resume is stored with 4 metadata fields: `fit_level`, `writing_style`, `name`, `skills` (comma-separated). ChromaDB's native `where` clause filters at the index level, so the API endpoint passes `where={"fit_level": fit_level}` straight through with no post-processing. The collection uses `metadata={"hnsw:space": "cosine"}` to match `all-MiniLM-L6-v2`'s training objective. Similarity scores are `1 - distance`, ranging from 0 (unrelated) to 1 (identical).

## Alternatives Considered

**FAISS (used in P2)** - The fastest approximate nearest neighbor library, purpose-built for billion-scale vector search. At 250 documents, FAISS's raw ANN throughput advantage is unmeasurable since both return results in <50ms. FAISS's problem for P4 is that it's an in-memory index with no persistence. `faiss.write_index()`/`faiss.read_index()` exist but require manual serialization of metadata alongside the index file, manual ID-to-document mapping, and manual rebuild logic. ChromaDB handles all of this as a database with `PersistentClient` persisting both vectors and metadata to SQLite automatically. For P2's batch pattern (build, evaluate, exit), FAISS's simplicity was an advantage. For P4's live API pattern (start, serve queries, restart, serve again), the lack of persistence is a dealbreaker.

**Pinecone** - Fully managed vector database with production-grade features: automatic scaling, backups, monitoring. But P4 has 250 documents. Pinecone's free tier would work, but it adds a network dependency (every query is an HTTP round-trip to a remote server) and requires an API key. The Streamlit demo and API must work offline for local demos.

**Qdrant** - Better production story than ChromaDB with distributed deployment, a Rust engine, and a more mature filtering DSL. But Qdrant requires either Docker (`qdrant/qdrant` container) or a separate server process. P4 runs as a single Python process (FastAPI or Streamlit). Adding a Docker dependency means anyone running the demo needs Docker installed. ChromaDB's `PersistentClient` embeds inside the Python process with no separate infrastructure.

**pgvector** - SQL + vector search in one database, using PostgreSQL's `VECTOR` type. The production choice for applications that already have PostgreSQL. But P4 has no database: all data lives in JSONL files on disk. Installing PostgreSQL, running migrations, and loading JSONL into tables to serve 250 vectors is the wrong tool for the job. pgvector solves "I already have Postgres, how do I add vectors?" P4 has the opposite problem: "I have vectors, I need the simplest possible store."

## Quantified Validation

API and Streamlit start instantly via `get_collection()` with no index rebuild and no model load. `search_similar("Python machine learning data science", top_k=5)` returns relevant candidates ranked by cosine similarity in <50ms. Metadata filtering works at the index level: `where={"fit_level": "excellent"}` narrows results to the ~50 excellent-fit resumes without touching the other 200. `collection.count()` powers the `/health` endpoint's `vector_store_ready` field and the search response's `total_in_index` count. The 2-second startup penalty from P2's FAISS pattern is eliminated.

## Consequences

`fit_level` filtering is a single `where` parameter on the query call, passed straight through from the API endpoint to ChromaDB's HNSW index with no Python-side post-filtering. `collection.count()` provides direct database introspection that FAISS would require manual tracking to replicate.

ChromaDB 1.5's `EmbeddingFunction` conflict required the `query_embeddings` workaround, 5 extra lines in `search_similar()` to manage the EF singleton manually. The collection must be deleted and recreated on re-index (`build_resume_index()` calls `client.delete_collection()` then `client.create_collection()`) to avoid duplicate IDs, since there's no upsert-on-conflict for the full-rebuild case. ChromaDB's `PersistentClient` uses SQLite under the hood, which is not suitable for multi-process concurrent writes. This is fine for P4's single-writer (pipeline builds index) / multi-reader (API + Streamlit query it) pattern, but would need Qdrant or Pinecone for true multi-writer production use. ChromaDB also adds a transitive dependency on `onnxruntime` (~50MB) even though P4 uses its own SentenceTransformer for encoding; the default embedding model is never used but the dependency is pulled in.

P2 chose FAISS because it needed raw ANN speed and direct index control for evaluating retrieval quality, where persistence didn't matter. When P4's requirements shifted to live serving with persistence and metadata filtering, reusing FAISS would have meant building a persistence layer, a metadata store, and a query API on top of it. (This is the same progression as moving from an in-memory `ConcurrentHashMap` cache to an embedded H2 database in Java when your service needs to survive restarts.)
