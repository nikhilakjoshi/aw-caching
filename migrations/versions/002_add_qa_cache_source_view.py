"""Add qa_cache_source view

Revision ID: 002_add_qa_cache_source_view
Revises: 001_add_qa_cache_embeddings
Create Date: 2025-09-15 10:05:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "002_add_qa_cache_source_view"
down_revision: Union[str, None] = "001_add_qa_cache_embeddings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create qa_cache_source view."""
    op.execute(
        """
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
    """
    )


def downgrade() -> None:
    """Drop qa_cache_source view."""
    op.execute("DROP VIEW IF EXISTS askwealth.qa_cache_source;")
