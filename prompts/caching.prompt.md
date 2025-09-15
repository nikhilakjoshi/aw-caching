# Requirements Document: QnA Caching Module with Embeddings

## Objective

Implement a **caching module** in the FastAPI QnA service that prevents redundant LLM calls. The cache will store previously answered questions (with embeddings) and return answers from cache when similar questions are asked again.

The solution must use:

- **LlamaIndex** for embeddings and similarity search.
- **Postgres (askwealth schema)** for persistence.
- **Alembic + SQLAlchemy** for all schema migrations (format must match existing migrations, see attached `config` table example).

---

## Current QnA Flow

```
User Question
   ↓
Vagueness Check
   ↓
Rephrase Question (small LLM call)
   ↓
RAG (vector search + LLM)
   ↓
Answer Returned
```

---

## New QnA Flow (with Cache)

```
User Question
   ↓
Vagueness Check
   ↓
Rephrase Question (small LLM call)
   ↓
Cache Lookup
   ├─ Cache Hit → Return Cached Answer
   └─ Cache Miss → RAG (vector search + LLM) → Answer → Cache Update
```

---

## Components to Build

### 1. `QACache` Class

A self-contained drop-in Python module (`qa_cache.py`) to be injected at the FastAPI endpoint level.

#### Responsibilities

- **Lookup**

  - Input: `rephrased_question`
  - Generate embedding (via LlamaIndex)
  - Perform similarity search against cache embeddings
  - If similarity ≥ threshold → return cached answer

- **Update**

  - After cache miss → save `(original_question, rephrased_question, answer, embedding)` into DB

- **Clear**

  - Optional admin/debug operation to clear cache

#### Configurations

- `similarity_threshold` (default: 0.85)
- `embedding_model` (any LlamaIndex model)
- `persist_path` (optional for local dev)

#### Interface

```python
class QACache:
    def __init__(self, embedding_model, threshold=0.85, vector_store=None):
        ...

    async def lookup(self, rephrased_question: str) -> Optional[str]:
        """Return cached answer if similar enough, else None."""

    async def update(self, original_question: str, rephrased_question: str, answer: str):
        """Insert new Q/A pair into cache."""

    async def clear(self):
        """Clear cache entries (admin/debug)."""
```

---

### 2. FastAPI Integration

`QACache` should be used as a dependency inside the QnA endpoint, not at startup.

#### Example Endpoint

```python
@app.post("/qna", response_model=AnswerResponse)
async def qna_endpoint(req: QuestionRequest, cache: QACache = Depends(get_cache)):
    if not await vagueness_check(req.question):
        return AnswerResponse(answer="Too vague", source="system")

    rephrased = await rephrase_question(req.question)

    cached_answer = await cache.lookup(rephrased)
    if cached_answer:
        return AnswerResponse(answer=cached_answer, source="cache")

    rag_answer = await do_rag(rephrased)

    await cache.update(req.question, rephrased, rag_answer)

    return AnswerResponse(answer=rag_answer, source="rag")
```

---

### 3. Postgres Schema

#### 3.1 View: `qa_cache_source`

Pairs user messages with their assistant replies.
Pulls `original_question` and `rephrased_question` from annotations.

```sql
CREATE OR REPLACE VIEW askwealth.qa_cache_source AS
SELECT
    u.thread_id,
    u.message_id AS user_message_id,
    u.parts AS user_parts,
    u.annotations AS user_annotations,
    a.message_id AS assistant_message_id,
    a.parts AS assistant_parts,
    a.annotations AS assistant_annotations,
    a.annotations->'debug'->'relevant_rephrase_question_result'->>'original_question' AS original_question,
    a.annotations->'debug'->'relevant_rephrase_question_result'->>'rephrased_question' AS rephrased_question
FROM askwealth.message u
JOIN askwealth.message a
  ON u.thread_id = a.thread_id
 AND a.role = 'assistant'
 AND u.role = 'user'
 AND a.created_at = (
        SELECT MIN(created_at)
        FROM askwealth.message
        WHERE thread_id = u.thread_id
          AND role = 'assistant'
          AND created_at > u.created_at
    )
WHERE a.annotations->'debug'->'relevant_rephrase_question_result'->>'relevant' = 'true';
```

---

#### 3.2 Table: `qa_cache_embeddings`

Stores embeddings + references to Q\&A pairs.

```sql
CREATE TABLE askwealth.qa_cache_embeddings (
    embedding_id uuid NOT NULL,
    source_user_message_id uuid NOT NULL,
    source_assistant_message_id uuid NOT NULL,
    rephrased_question text NOT NULL,
    embedding vector NOT NULL,
    created_at timestamp NOT NULL,
    updated_at timestamp NOT NULL,
    created_by varchar(50) NOT NULL,
    updated_by varchar(50) NOT NULL,
    CONSTRAINT qa_cache_embeddings_pkey PRIMARY KEY (embedding_id),
    CONSTRAINT qa_cache_embeddings_user_fkey FOREIGN KEY (source_user_message_id)
        REFERENCES askwealth.message (message_id) ON DELETE CASCADE,
    CONSTRAINT qa_cache_embeddings_assistant_fkey FOREIGN KEY (source_assistant_message_id)
        REFERENCES askwealth.message (message_id) ON DELETE CASCADE
);
```

---

### 4. Alembic Migration Format

All migrations must follow the **same format** as existing migrations (`alembic + sqlalchemy`, see `config` migration example).

#### Example: Migration for `qa_cache_embeddings`

```python
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "XXXX_add_qa_cache_embeddings"
down_revision: Union[str, None] = "<<<PREVIOUS_REVISION>>>"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "qa_cache_embeddings",
        sa.Column("embedding_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_user_message_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_assistant_message_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rephrased_question", sa.Text(), nullable=False),
        sa.Column("embedding", postgresql.ARRAY(sa.Float()), nullable=False),  # Or vector if pgvector
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=False),
        sa.Column("created_by", sa.String(length=50), nullable=False),
        sa.Column("updated_by", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("embedding_id"),
        sa.ForeignKeyConstraint(["source_user_message_id"], ["askwealth.message.message_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_assistant_message_id"], ["askwealth.message.message_id"], ondelete="CASCADE"),
        schema="askwealth",
    )


def downgrade() -> None:
    op.drop_table("qa_cache_embeddings", schema="askwealth")
```

#### Example: Migration for `qa_cache_source` (view)

```python
def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE VIEW askwealth.qa_cache_source AS
        SELECT
            u.thread_id,
            u.message_id AS user_message_id,
            u.parts AS user_parts,
            u.annotations AS user_annotations,
            a.message_id AS assistant_message_id,
            a.parts AS assistant_parts,
            a.annotations AS assistant_annotations,
            a.annotations->'debug'->'relevant_rephrase_question_result'->>'original_question' AS original_question,
            a.annotations->'debug'->'relevant_rephrase_question_result'->>'rephrased_question' AS rephrased_question
        FROM askwealth.message u
        JOIN askwealth.message a
          ON u.thread_id = a.thread_id
         AND a.role = 'assistant'
         AND u.role = 'user'
         AND a.created_at = (
                SELECT MIN(created_at)
                FROM askwealth.message
                WHERE thread_id = u.thread_id
                  AND role = 'assistant'
                  AND created_at > u.created_at
            )
        WHERE a.annotations->'debug'->'relevant_rephrase_question_result'->>'relevant' = 'true';
    """)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS askwealth.qa_cache_source;")
```

---

### 5. Monitoring

- Log **cache hits vs misses**.
- Track row counts in `qa_cache_embeddings`.
- Optional `/cache/stats` endpoint for metrics.

---

## Deliverables

1. `qa_cache.py` with the `QACache` class (LlamaIndex integration).
2. Alembic migration file for `qa_cache_embeddings` table (format as above).
3. Alembic migration file for `qa_cache_source` view (format as above).
4. Example FastAPI integration (`/qna` endpoint).
5. Documentation on configuration options.
