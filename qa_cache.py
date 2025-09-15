"""
QnA Caching Module with Embeddings

A self-contained caching module for FastAPI QnA service that prevents redundant LLM calls
by storing previously answered questions with embeddings and returning cached answers
when similar questions are asked again.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select, insert, delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import Column, String, Text, TIMESTAMP, ForeignKey, Table, MetaData
from sqlalchemy.dialects.postgresql import ARRAY, REAL
from sqlalchemy.orm import declarative_base

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.schema import TextNode


Base = declarative_base()


class QACacheEmbeddings(Base):
    """SQLAlchemy model for qa_cache_embeddings table."""

    __tablename__ = "qa_cache_embeddings"
    __table_args__ = {"schema": "askwealth"}

    embedding_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_user_message_id = Column(UUID(as_uuid=True), nullable=False)
    source_assistant_message_id = Column(UUID(as_uuid=True), nullable=False)
    rephrased_question = Column(Text, nullable=False)
    embedding = Column(ARRAY(REAL), nullable=False)
    created_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)
    created_by = Column(String(50), nullable=False, default="system")
    updated_by = Column(String(50), nullable=False, default="system")


@dataclass
class CacheResult:
    """Result from cache lookup."""

    answer: str
    similarity_score: float
    source_user_message_id: str
    source_assistant_message_id: str


class QACache:
    """
    QnA Cache with embeddings for similarity-based lookup.

    Provides lookup, update, and clear operations for cached Q&A pairs.
    Uses LlamaIndex for embeddings and similarity search.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        embedding_model=None,
        similarity_threshold: float = 0.85,
        embedding_dim: int = 1536,
        created_by: str = "system",
    ):
        """
        Initialize QACache.

        Args:
            db_session: SQLAlchemy async session for database operations
            embedding_model: LlamaIndex embedding model (defaults to OpenAI)
            similarity_threshold: Minimum similarity score for cache hits (0.0-1.0)
            embedding_dim: Dimension of embedding vectors
            created_by: User identifier for audit trails
        """
        self.db_session = db_session
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim
        self.created_by = created_by

        # Initialize embedding model
        if embedding_model is None:
            self.embedding_model = OpenAIEmbedding(
                model="text-embedding-3-small", embed_batch_size=100
            )
        else:
            self.embedding_model = embedding_model

        # Set global settings for LlamaIndex
        Settings.embed_model = self.embedding_model

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        embedding = await self.embedding_model.aget_text_embedding(text)
        return embedding

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return float(dot_product / (norm_v1 * norm_v2))

    async def lookup(self, rephrased_question: str) -> Optional[CacheResult]:
        """
        Look up cached answer for similar question.

        Args:
            rephrased_question: The rephrased question to search for

        Returns:
            CacheResult if similar question found above threshold, None otherwise
        """
        # Generate embedding for the question
        question_embedding = await self._generate_embedding(rephrased_question)

        # Fetch all cached embeddings
        stmt = select(QACacheEmbeddings)
        result = await self.db_session.execute(stmt)
        cached_entries = result.scalars().all()

        if not cached_entries:
            return None

        # Find most similar cached question
        best_similarity = 0.0
        best_entry = None

        for entry in cached_entries:
            similarity = self._cosine_similarity(question_embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_entry = entry

        # Check if similarity meets threshold
        if best_similarity < self.similarity_threshold:
            return None

        # Fetch the assistant's answer from the message table
        answer_query = text(
            """
            SELECT parts 
            FROM askwealth.message 
            WHERE message_id = :message_id
        """
        )

        result = await self.db_session.execute(
            answer_query, {"message_id": best_entry.source_assistant_message_id}
        )
        answer_row = result.fetchone()

        if not answer_row:
            return None

        # Extract answer text from parts (assuming it's a JSON array)
        answer_parts = answer_row[0]
        if isinstance(answer_parts, list) and len(answer_parts) > 0:
            answer = (
                answer_parts[0].get("text", "")
                if isinstance(answer_parts[0], dict)
                else str(answer_parts[0])
            )
        else:
            answer = str(answer_parts)

        return CacheResult(
            answer=answer,
            similarity_score=best_similarity,
            source_user_message_id=str(best_entry.source_user_message_id),
            source_assistant_message_id=str(best_entry.source_assistant_message_id),
        )

    async def update(
        self,
        original_question: str,
        rephrased_question: str,
        answer: str,
        user_message_id: Optional[str] = None,
        assistant_message_id: Optional[str] = None,
    ) -> str:
        """
        Add new Q&A pair to cache.

        Args:
            original_question: Original user question
            rephrased_question: Rephrased/cleaned question
            answer: LLM-generated answer
            user_message_id: UUID of user message (optional)
            assistant_message_id: UUID of assistant message (optional)

        Returns:
            UUID of the created cache entry
        """
        # Generate embedding for rephrased question
        embedding = await self._generate_embedding(rephrased_question)

        # Create cache entry
        cache_entry = QACacheEmbeddings(
            embedding_id=uuid.uuid4(),
            source_user_message_id=(
                uuid.uuid4() if user_message_id is None else uuid.UUID(user_message_id)
            ),
            source_assistant_message_id=(
                uuid.uuid4()
                if assistant_message_id is None
                else uuid.UUID(assistant_message_id)
            ),
            rephrased_question=rephrased_question,
            embedding=embedding,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by=self.created_by,
            updated_by=self.created_by,
        )

        # Insert into database
        self.db_session.add(cache_entry)
        await self.db_session.commit()

        return str(cache_entry.embedding_id)

    async def clear(self, confirm: bool = False) -> int:
        """
        Clear all cache entries.

        Args:
            confirm: Must be True to actually clear the cache

        Returns:
            Number of entries deleted
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to clear cache")

        # Count existing entries
        count_stmt = select(QACacheEmbeddings)
        result = await self.db_session.execute(count_stmt)
        count = len(result.scalars().all())

        # Delete all entries
        delete_stmt = delete(QACacheEmbeddings)
        await self.db_session.execute(delete_stmt)
        await self.db_session.commit()

        return count

    async def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        # Count total entries
        count_stmt = select(QACacheEmbeddings)
        result = await self.db_session.execute(count_stmt)
        total_entries = len(result.scalars().all())

        # Get date range
        date_range_query = text(
            """
            SELECT 
                MIN(created_at) as earliest,
                MAX(created_at) as latest
            FROM askwealth.qa_cache_embeddings
        """
        )

        result = await self.db_session.execute(date_range_query)
        date_row = result.fetchone()

        return {
            "total_entries": total_entries,
            "earliest_entry": date_row[0] if date_row and date_row[0] else None,
            "latest_entry": date_row[1] if date_row and date_row[1] else None,
            "similarity_threshold": self.similarity_threshold,
            "embedding_dimension": self.embedding_dim,
        }


# Dependency function for FastAPI
async def get_qa_cache(db_session: AsyncSession) -> QACache:
    """
    FastAPI dependency function to get QACache instance.

    Args:
        db_session: Database session dependency

    Returns:
        Configured QACache instance
    """
    return QACache(
        db_session=db_session,
        similarity_threshold=0.85,  # Can be configured via environment
        created_by="api",
    )
