# QnA Caching Module Documentation

## Overview

The QnA Caching Module is a drop-in solution for FastAPI applications that prevents redundant LLM calls by caching previously answered questions with embeddings and returning cached answers when similar questions are asked again.

## Features

- **Embedding-based similarity search** using LlamaIndex
- **PostgreSQL persistence** with proper schema management
- **Async support** throughout the entire pipeline
- **FastAPI integration** with dependency injection
- **Configurable similarity thresholds**
- **Comprehensive monitoring and statistics**
- **Alembic migrations** for schema management

## Installation

### Dependencies

Add the following to your `requirements.txt`:

```txt
# Core dependencies
fastapi>=0.104.0
uvicorn>=0.24.0
sqlalchemy>=2.0.0
asyncpg>=0.29.0
alembic>=1.13.0
pydantic>=2.0.0

# LlamaIndex dependencies
llama-index-core>=0.10.0
llama-index-embeddings-openai>=0.1.0

# Additional dependencies
numpy>=1.24.0
python-multipart>=0.0.6
```

### Database Setup

1. Ensure PostgreSQL is running with the `askwealth` schema
2. Install pgvector extension (optional, but recommended for better vector performance):

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. Run Alembic migrations:
   ```bash
   alembic upgrade head
   ```

## Configuration

### Environment Variables

Set the following environment variables:

```bash
# Database connection
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/dbname

# OpenAI API (for embeddings)
OPENAI_API_KEY=your_openai_api_key

# Cache configuration (optional)
QA_CACHE_SIMILARITY_THRESHOLD=0.85
QA_CACHE_EMBEDDING_MODEL=text-embedding-3-small
QA_CACHE_CREATED_BY=api
```

### QACache Configuration Options

| Parameter              | Type             | Default                       | Description                                        |
| ---------------------- | ---------------- | ----------------------------- | -------------------------------------------------- |
| `similarity_threshold` | float            | 0.85                          | Minimum cosine similarity for cache hits (0.0-1.0) |
| `embedding_model`      | LlamaIndex Model | OpenAI text-embedding-3-small | Embedding model for question encoding              |
| `embedding_dim`        | int              | 1536                          | Dimension of embedding vectors                     |
| `created_by`           | str              | "system"                      | User identifier for audit trails                   |

## Usage

### Basic Integration

```python
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from qa_cache import QACache, get_qa_cache

app = FastAPI()

@app.post("/qna")
async def qna_endpoint(
    question: str,
    cache: QACache = Depends(get_qa_cache),
    db_session: AsyncSession = Depends(get_db_session)
):
    # 1. Check cache
    cached_answer = await cache.lookup(question)
    if cached_answer:
        return {"answer": cached_answer.answer, "source": "cache"}

    # 2. Generate answer (your RAG pipeline)
    answer = await your_rag_pipeline(question)

    # 3. Update cache
    await cache.update("original_question", question, answer)

    return {"answer": answer, "source": "rag"}
```

### Advanced Integration

See `fastapi_example.py` for a complete implementation with:

- Vagueness checking
- Question rephrasing
- Proper error handling
- Response time tracking
- Cache statistics endpoints

## API Endpoints

### Main QnA Endpoint

**POST `/qna`**

Request:

```json
{
  "question": "What is the capital of France?",
  "user_id": "optional_user_id"
}
```

Response:

```json
{
  "answer": "The capital of France is Paris.",
  "source": "cache|rag|system",
  "similarity_score": 0.92,
  "response_time_ms": 45.2,
  "cached_entry_id": "uuid"
}
```

### Cache Management

**GET `/cache/stats`** - Get cache statistics

```json
{
  "total_entries": 150,
  "earliest_entry": "2025-09-01T10:00:00Z",
  "latest_entry": "2025-09-15T15:30:00Z",
  "similarity_threshold": 0.85,
  "embedding_dimension": 1536
}
```

**DELETE `/cache/clear?confirm=true`** - Clear all cache entries

```json
{
  "message": "Successfully cleared 150 cache entries",
  "deleted_count": 150
}
```

## Database Schema

### Tables

#### `askwealth.qa_cache_embeddings`

Stores question embeddings and references to Q&A pairs.

| Column                        | Type        | Description                    |
| ----------------------------- | ----------- | ------------------------------ |
| `embedding_id`                | UUID        | Primary key                    |
| `source_user_message_id`      | UUID        | Reference to user message      |
| `source_assistant_message_id` | UUID        | Reference to assistant message |
| `rephrased_question`          | TEXT        | Processed question text        |
| `embedding`                   | FLOAT[]     | Question embedding vector      |
| `created_at`                  | TIMESTAMP   | Creation timestamp             |
| `updated_at`                  | TIMESTAMP   | Last update timestamp          |
| `created_by`                  | VARCHAR(50) | Creator identifier             |
| `updated_by`                  | VARCHAR(50) | Last updater identifier        |

#### Foreign Key Constraints

- `source_user_message_id` → `askwealth.message.message_id` (CASCADE)
- `source_assistant_message_id` → `askwealth.message.message_id` (CASCADE)

### Views

#### `askwealth.qa_cache_source`

Materialized view that pairs user questions with assistant answers, extracting original and rephrased questions from annotations.

## Performance Considerations

### Similarity Search Optimization

- **Threshold tuning**: Lower thresholds (0.8) increase cache hits but may reduce answer quality
- **Embedding model**: Smaller models (text-embedding-3-small) are faster but less accurate
- **Batch processing**: Consider implementing batch embedding for bulk operations

### Database Optimization

- **Indexes**: Created automatically on `rephrased_question`, `created_at`, and message IDs
- **Vector storage**: Consider pgvector extension for large-scale deployments
- **Connection pooling**: Use proper connection pooling for high-traffic scenarios

### Memory Usage

- **Embedding cache**: Consider implementing in-memory LRU cache for frequently accessed embeddings
- **Batch size**: Adjust LlamaIndex `embed_batch_size` based on available memory

## Monitoring

### Logging

The module logs the following events:

- Cache hits/misses with similarity scores
- Cache updates with entry IDs
- Error conditions and recovery

### Metrics to Track

- **Cache hit ratio**: Percentage of questions served from cache
- **Average similarity scores**: Quality of cache matches
- **Response times**: Cache vs RAG response times
- **Cache growth**: Rate of new entries added

### Example Monitoring

```python
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Track cache performance
cache_hits = 0
total_requests = 0

@app.middleware("http")
async def track_cache_performance(request, call_next):
    global cache_hits, total_requests

    response = await call_next(request)

    if request.url.path == "/qna":
        total_requests += 1
        if response.headers.get("x-cache-source") == "cache":
            cache_hits += 1

    return response
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: ModuleNotFoundError for LlamaIndex or SQLAlchemy
**Solution**: Install dependencies with `pip install -r requirements.txt`

#### 2. Database Connection Issues

**Problem**: asyncpg connection errors
**Solution**:

- Verify DATABASE_URL format
- Ensure PostgreSQL is running
- Check network connectivity and credentials

#### 3. Embedding Generation Errors

**Problem**: OpenAI API errors or rate limits
**Solution**:

- Verify OPENAI_API_KEY is set
- Implement retry logic with exponential backoff
- Consider switching to local embedding models

#### 4. Low Cache Hit Rates

**Problem**: Most questions result in cache misses
**Solutions**:

- Lower similarity threshold (0.75-0.8)
- Improve question preprocessing/normalization
- Verify embedding model consistency
- Check for data quality issues

### Performance Issues

#### 1. Slow Similarity Search

**Solutions**:

- Enable pgvector extension
- Add database indexes
- Implement approximate nearest neighbor search

#### 2. High Memory Usage

**Solutions**:

- Reduce embedding batch size
- Implement connection pooling
- Use smaller embedding models

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger("qa_cache").setLevel(logging.DEBUG)
logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
```

## Migration Guide

### Running Migrations

```bash
# Initialize Alembic (if not already done)
alembic init migrations

# Generate new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback if needed
alembic downgrade -1
```

### Custom Migration Template

For consistency with existing migrations, use this template:

```python
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'XXX_description'
down_revision: Union[str, None] = 'previous_revision'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Your upgrade logic
    pass

def downgrade() -> None:
    # Your downgrade logic
    pass
```

## Security Considerations

### Data Privacy

- **PII Handling**: Ensure questions don't contain sensitive information
- **Access Control**: Implement proper authentication/authorization
- **Audit Trails**: Use `created_by`/`updated_by` fields for tracking

### API Security

- **Rate Limiting**: Implement rate limiting for cache endpoints
- **Input Validation**: Validate question length and content
- **CORS**: Configure appropriate CORS policies

## Best Practices

1. **Question Preprocessing**: Normalize questions before caching
2. **Cache Warming**: Pre-populate cache with common questions
3. **Regular Cleanup**: Implement periodic cleanup of old cache entries
4. **Monitoring**: Set up alerts for cache hit rates and errors
5. **Testing**: Include cache behavior in your test suite

## Example Test Suite

```python
import pytest
from qa_cache import QACache

@pytest.mark.asyncio
async def test_cache_lookup():
    cache = QACache(db_session, similarity_threshold=0.8)

    # Test cache miss
    result = await cache.lookup("What is AI?")
    assert result is None

    # Add to cache
    await cache.update("What is AI?", "What is AI?", "AI is...")

    # Test cache hit
    result = await cache.lookup("What is artificial intelligence?")
    assert result is not None
    assert result.similarity_score > 0.8
```

## Support

For issues or questions:

1. Check the troubleshooting section
2. Review logs for error details
3. Verify configuration and dependencies
4. Test with minimal examples

## License

This module is part of the AskWealth platform and follows the same licensing terms.
