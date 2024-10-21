"""Update relationships between Video and ChatMessage

Revision ID: 3094ee78171d
Revises: 15f8b737935d
Create Date: 2024-10-20 23:40:53.988954

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3094ee78171d'
down_revision: Union[str, None] = '15f8b737935d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('videos', sa.Column('chat_message_id', sa.Integer(), nullable=True))
    op.create_foreign_key(None, 'videos', 'chat_messages', ['chat_message_id'], ['id'], ondelete='CASCADE')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'videos', type_='foreignkey')
    op.drop_column('videos', 'chat_message_id')
    # ### end Alembic commands ###
