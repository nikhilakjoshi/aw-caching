"""
FastAPI Integration Example for QnA Caching Module

This file demonstrates how to integrate the QACache class into a FastAPI application
with proper dependency injection and caching flow.
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from qa_cache import QACache, get_qa_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="QnA Service with Caching",
    description="FastAPI service for Q&A with intelligent caching using embeddings",
    version="1.0.0",
)

# Database configuration (example - adjust for your setup)
DATABASE_URL = "postgresql+asyncpg://user:password@localhost/dbname"
engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


# Pydantic models
class QuestionRequest(BaseModel):
    """Request model for questions."""

    question: str = Field(
        ..., min_length=1, max_length=1000, description="User question"
    )
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class AnswerResponse(BaseModel):
    """Response model for answers."""

    answer: str = Field(..., description="Generated or cached answer")
    source: str = Field(
        ..., description="Source of answer: 'cache', 'rag', or 'system'"
    )
    similarity_score: Optional[float] = Field(
        None, description="Similarity score if from cache"
    )
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    cached_entry_id: Optional[str] = Field(
        None, description="Cache entry ID if from cache"
    )


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""

    total_entries: int
    earliest_entry: Optional[datetime]
    latest_entry: Optional[datetime]
    similarity_threshold: float
    embedding_dimension: int


# Database dependency
async def get_db_session() -> AsyncSession:
    """Get database session dependency."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


# QACache dependency that uses the database session
async def get_cache_with_session(
    db_session: AsyncSession = Depends(get_db_session),
) -> QACache:
    """Get QACache instance with database session."""
    return QACache(
        db_session=db_session,
        similarity_threshold=0.85,  # Can be configured via environment
        created_by="api",
    )


# Mock functions (replace with your actual implementations)
async def vagueness_check(question: str) -> bool:
    """
    Check if question is too vague.

    Args:
        question: User question

    Returns:
        True if question is specific enough, False if too vague
    """
    # Simple mock implementation
    return len(question.strip()) > 10 and "?" in question


async def rephrase_question(question: str) -> str:
    """
    Rephrase question for better processing.

    Args:
        question: Original question

    Returns:
        Rephrased question
    """
    # Mock implementation - replace with actual LLM call
    rephrased = question.strip().lower()
    if not rephrased.endswith("?"):
        rephrased += "?"
    return rephrased


async def do_rag(rephrased_question: str) -> str:
    """
    Perform RAG (Retrieval Augmented Generation) to generate answer.

    Args:
        rephrased_question: Cleaned/rephrased question

    Returns:
        Generated answer
    """
    # Mock implementation - replace with actual RAG pipeline
    await asyncio.sleep(0.1)  # Simulate processing time
    return f"This is a generated answer for: {rephrased_question}"


# Main QnA endpoint
@app.post("/qna", response_model=AnswerResponse)
async def qna_endpoint(
    req: QuestionRequest,
    cache: QACache = Depends(get_cache_with_session),
) -> AnswerResponse:
    """
    Main QnA endpoint with caching support.

    Flow:
    1. Vagueness check
    2. Rephrase question
    3. Cache lookup
    4. If cache miss: RAG + cache update
    5. Return answer with metadata
    """
    start_time = datetime.utcnow()

    try:
        # Step 1: Vagueness check
        if not await vagueness_check(req.question):
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return AnswerResponse(
                answer="Your question seems too vague. Please provide more specific details.",
                source="system",
                response_time_ms=response_time,
            )

        # Step 2: Rephrase question
        rephrased = await rephrase_question(req.question)
        logger.info(f"Rephrased question: '{req.question}' -> '{rephrased}'")

        # Step 3: Cache lookup
        cache_result = await cache.lookup(rephrased)

        if cache_result:
            # Cache hit
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"Cache hit with similarity {cache_result.similarity_score:.3f}"
            )

            return AnswerResponse(
                answer=cache_result.answer,
                source="cache",
                similarity_score=cache_result.similarity_score,
                response_time_ms=response_time,
                cached_entry_id=cache_result.source_assistant_message_id,
            )

        # Step 4: Cache miss - perform RAG
        logger.info("Cache miss - performing RAG")
        rag_answer = await do_rag(rephrased)

        # Step 5: Update cache
        cache_entry_id = await cache.update(
            original_question=req.question,
            rephrased_question=rephrased,
            answer=rag_answer,
        )

        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"RAG completed and cached with ID: {cache_entry_id}")

        return AnswerResponse(
            answer=rag_answer,
            source="rag",
            response_time_ms=response_time,
            cached_entry_id=cache_entry_id,
        )

    except Exception as e:
        logger.error(f"Error in QnA endpoint: {str(e)}")
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return AnswerResponse(
            answer="I apologize, but I encountered an error processing your question.",
            source="system",
            response_time_ms=response_time,
        )


# Cache management endpoints
@app.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats(
    cache: QACache = Depends(get_cache_with_session),
) -> CacheStatsResponse:
    """Get cache statistics."""
    try:
        stats = await cache.stats()
        return CacheStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics",
        )


@app.delete("/cache/clear")
async def clear_cache(
    confirm: bool = False, cache: QACache = Depends(get_cache_with_session)
) -> dict:
    """
    Clear all cache entries.

    Args:
        confirm: Must be True to actually clear the cache

    Returns:
        Dictionary with number of deleted entries
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must pass confirm=true to clear cache",
            )

        deleted_count = await cache.clear(confirm=True)
        logger.info(f"Cleared {deleted_count} cache entries")

        return {
            "message": f"Successfully cleared {deleted_count} cache entries",
            "deleted_count": deleted_count,
        }

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache",
        )


# Health check endpoint
@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "qna-caching-service",
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("QnA Caching Service starting up...")
    # Add any initialization logic here


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("QnA Caching Service shutting down...")
    await engine.dispose()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_example:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
