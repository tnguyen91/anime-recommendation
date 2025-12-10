"""Add unique constraint on user_favorites

Revision ID: 5a2b3c4d5e6f
Revises: 44c615c39c5f
Create Date: 2025-12-03

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '5a2b3c4d5e6f'
down_revision = '44c615c39c5f'
branch_labels = None
depends_on = None


def upgrade():
    # Add unique constraint on (user_id, anime_id) to prevent duplicate favorites
    op.create_unique_constraint('uix_user_anime', 'user_favorites', ['user_id', 'anime_id'])


def downgrade():
    op.drop_constraint('uix_user_anime', 'user_favorites', type_='unique')
