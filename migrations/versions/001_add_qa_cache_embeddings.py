"""Add qa_cache_embeddings table

Revision ID: 001_add_qa_cache_embeddings
Revises:
Create Date: 2025-09-15 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_add_qa_cache_embeddings"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create qa_cache_embeddings table."""
    op.create_table(
        "qa_cache_embeddings",
        sa.Column("embedding_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "source_user_message_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column(
            "source_assistant_message_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column("rephrased_question", sa.Text(), nullable=False),
        sa.Column("embedding", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(), nullable=False),
        sa.Column("created_by", sa.String(length=50), nullable=False),
        sa.Column("updated_by", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("embedding_id"),
        sa.ForeignKeyConstraint(
            ["source_user_message_id"],
            ["askwealth.message.message_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_assistant_message_id"],
            ["askwealth.message.message_id"],
            ondelete="CASCADE",
        ),
        schema="askwealth",
    )

    # Create index on rephrased_question for faster text searches
    op.create_index(
        "ix_qa_cache_embeddings_rephrased_question",
        "qa_cache_embeddings",
        ["rephrased_question"],
        schema="askwealth",
    )

    # Create index on created_at for time-based queries
    op.create_index(
        "ix_qa_cache_embeddings_created_at",
        "qa_cache_embeddings",
        ["created_at"],
        schema="askwealth",
    )

    # Create composite index on message IDs for join performance
    op.create_index(
        "ix_qa_cache_embeddings_message_ids",
        "qa_cache_embeddings",
        ["source_user_message_id", "source_assistant_message_id"],
        schema="askwealth",
    )


def downgrade() -> None:
    """Drop qa_cache_embeddings table."""
    op.drop_index(
        "ix_qa_cache_embeddings_message_ids",
        table_name="qa_cache_embeddings",
        schema="askwealth",
    )
    op.drop_index(
        "ix_qa_cache_embeddings_created_at",
        table_name="qa_cache_embeddings",
        schema="askwealth",
    )
    op.drop_index(
        "ix_qa_cache_embeddings_rephrased_question",
        table_name="qa_cache_embeddings",
        schema="askwealth",
    )
    op.drop_table("qa_cache_embeddings", schema="askwealth")
