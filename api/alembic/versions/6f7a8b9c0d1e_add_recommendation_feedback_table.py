"""Add recommendation feedback table

Revision ID: 6f7a8b9c0d1e
Revises: 5a2b3c4d5e6f
Create Date: 2026-01-05 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6f7a8b9c0d1e'
down_revision: Union[str, Sequence[str], None] = '5a2b3c4d5e6f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create recommendation_feedback table for tracking user interactions."""
    op.create_table(
        'recommendation_feedback',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('anime_id', sa.Integer(), nullable=False),
        sa.Column('action', sa.String(length=50), nullable=False),
        sa.Column('recommendation_request_id', sa.String(length=50), nullable=True),
        sa.Column('recorded_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_recommendation_feedback_id'),
        'recommendation_feedback',
        ['id'],
        unique=False
    )
    op.create_index(
        op.f('ix_recommendation_feedback_user_id'),
        'recommendation_feedback',
        ['user_id'],
        unique=False
    )
    op.create_index(
        op.f('ix_recommendation_feedback_anime_id'),
        'recommendation_feedback',
        ['anime_id'],
        unique=False
    )
    op.create_index(
        op.f('ix_recommendation_feedback_action'),
        'recommendation_feedback',
        ['action'],
        unique=False
    )
    op.create_index(
        op.f('ix_recommendation_feedback_recorded_at'),
        'recommendation_feedback',
        ['recorded_at'],
        unique=False
    )


def downgrade() -> None:
    """Drop recommendation_feedback table."""
    op.drop_index(op.f('ix_recommendation_feedback_recorded_at'), table_name='recommendation_feedback')
    op.drop_index(op.f('ix_recommendation_feedback_action'), table_name='recommendation_feedback')
    op.drop_index(op.f('ix_recommendation_feedback_anime_id'), table_name='recommendation_feedback')
    op.drop_index(op.f('ix_recommendation_feedback_user_id'), table_name='recommendation_feedback')
    op.drop_index(op.f('ix_recommendation_feedback_id'), table_name='recommendation_feedback')
    op.drop_table('recommendation_feedback')
