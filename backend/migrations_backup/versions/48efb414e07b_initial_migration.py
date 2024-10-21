"""Initial migration

Revision ID: 48efb414e07b
Revises: aa283f43bf27
Create Date: 2024-10-20 19:16:56.571748

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '48efb414e07b'
# down_revision: Union[str, None] = 'aa283f43bf27'

down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
