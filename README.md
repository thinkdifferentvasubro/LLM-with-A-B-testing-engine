# A/B Testing Agent with Long-Term Memory

An LLM-orchestrated A/B testing system: the LLM designs experiments and interprets results, a deterministic statistical engine runs the actual tests, and a RAG layer remembers past experiments per user.

```
User message
     │
     ▼
Intent Classifier ──► "chat" → direct response
     │
     ├──► conduct test   → A/B Test Agent ─┐
     ├──► retrieve tests → RAG Agent ───────┤
     └──► both ──────────────────────────────┤
                                              ▼
                                        Reasoning ──► Final response
```

Built with **LangGraph** (`src/llm/graph.py`), `InMemorySaver` checkpointer per thread.

1. **Intent classification** — routes to `chat` / `conduct test` / `retrieve tests` / `both`.
2. **Experiment design (LLM)** — given the CSV schema, fills a structured spec: episodes, metrics, covariates, tail direction. Infers sensible defaults when the user doesn't specify (natural allocation column, outcome-shaped metrics, pre-treatment covariates).
3. **Preprocessing (deterministic)** — dedup, mixed numeric/text extraction, date feature engineering, tiered missing-value handling (median/mode ≤15%, KNN 15–60%, drop >60%), IQR-based Winsorization, SMD covariate balance checks.
4. **Memory (RAG)** — every result gets embedded and stored per user for later retrieval.
5. **Reasoning** — final LLM pass turns raw stats into a plain-language answer.

## Statistical engine — `ABTestSelector`

Auto-selection is driven by per-group diagnostics: Shapiro-Wilk/D'Agostino normality, IQR outliers, and Levene's test for variance. A test only runs parametric if all groups are normal, the smallest group has ≥8 samples, and there are no outliers — otherwise it falls back nonparametric:

| Groups | Equal variance | Unequal variance | Nonparametric |
|---|---|---|---|
| 2 | Student's t-test | Welch's t-test | Mann-Whitney U |
| 3+ | One-way ANOVA | Alexander-Govern Welch ANOVA | Kruskal-Wallis |

Categorical metrics: chi-square, auto-downgrading to Fisher's exact when expected counts drop below 5. Multiple categorical metrics → log-linear GLM G-test. Multiple continuous metrics → MANOVA (Pillai's Trace), refused with a warning if normality fails. One-tailed and "test both directions" requests reuse the same decision logic with `alternative` swapped.

Users can also force a specific test (`run_selected_test`) — each test validates its own structural requirements (metric count, category count, tail support) and fails with a clear message rather than a stack trace. Diagnostics still run in this mode and come back as warnings (e.g. "you picked `ttest` but variances look unequal").

Every result is written as a natural-language summary — including the SMD covariate balance numbers — before being embedded into RAG memory.

## RAG memory — `VectorDBManager`

ChromaDB (`PersistentClient`) + a local `all-MiniLM-L6-v2` sentence-transformers model.

- **Self-bootstrapping model** — downloads and caches the embedding model on first run, then forces offline mode so nothing hits the network afterward.
- **Idempotent writes** — document ID is `sha256(user_id + document)`, so re-saving the same result is a no-op instead of a duplicate.
- **Per-user isolation** — every write/read is scoped by a `user_id` metadata filter at the Chroma level.
- **Storage** — single `my_collection`, persisted to `chroma_db/` on disk.

## Application layer

FastAPI + a Gradio UI mounted at `/ui`. Username/password auth (PBKDF2-HMAC-SHA256, salted) with JWT sessions (24h). Each login gets a fresh `thread_id`; CSVs go to a per-user uploads folder, wiped on shutdown. `users.json` and the default JWT secret are dev-grade — swap for a real store/secrets manager before production.

## Deployment

- **Tests** — `ab_bot_test.py` / `ab_rag_test.py` run as LangSmith experiments: real app output graded against a labeled dataset by an LLM-as-judge correctness evaluator, hard-failing on any mismatch.
- **CI** — both suites run on push, gating merges on zero correctness failures.
- **Runtime** — single ASGI process (FastAPI + Gradio), easy to containerize.
- **State to persist** — `chroma_db/` and `users.json` aren't backed by an external DB yet; `InMemorySaver` also means chat threads don't survive a restart.

## Tech stack

LangGraph · LangChain (Gemini 2.5 Flash) · scikit-learn / feature-engine / SciPy / statsmodels · FastAPI + Gradio · PyJWT · ChromaDB + sentence-transformers · LangSmith

## Status

Core pipeline and CI/CD test suites are complete. Still refining CI/CD and general polish.
