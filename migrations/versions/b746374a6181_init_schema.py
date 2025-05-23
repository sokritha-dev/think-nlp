"""init schema

Revision ID: b746374a6181
Revises: 
Create Date: 2025-05-08 08:25:12.244398

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b746374a6181'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('file_records',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('file_name', sa.String(), nullable=False),
    sa.Column('s3_key', sa.String(), nullable=False),
    sa.Column('s3_url', sa.String(), nullable=False),
    sa.Column('columns', sa.String(), nullable=False),
    sa.Column('record_count', sa.Integer(), nullable=False),
    sa.Column('file_hash', sa.String(), nullable=False),
    sa.Column('normalized_s3_key', sa.String(), nullable=True),
    sa.Column('normalized_s3_url', sa.String(), nullable=True),
    sa.Column('normalized_broken_map', sa.String(), nullable=True),
    sa.Column('normalized_updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('special_cleaned_s3_key', sa.String(), nullable=True),
    sa.Column('special_cleaned_s3_url', sa.String(), nullable=True),
    sa.Column('special_cleaned_flags', sa.String(), nullable=True),
    sa.Column('special_cleaned_removed', sa.String(), nullable=True),
    sa.Column('special_cleaned_updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('tokenized_s3_key', sa.String(), nullable=True),
    sa.Column('tokenized_s3_url', sa.String(), nullable=True),
    sa.Column('tokenized_updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('stopword_s3_key', sa.String(), nullable=True),
    sa.Column('stopword_s3_url', sa.String(), nullable=True),
    sa.Column('stopword_updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('stopword_config', sa.String(), nullable=True),
    sa.Column('lemmatized_s3_key', sa.String(), nullable=True),
    sa.Column('lemmatized_s3_url', sa.String(), nullable=True),
    sa.Column('lemmatized_updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('eda_wordcloud_url', sa.String(), nullable=True),
    sa.Column('eda_text_length_url', sa.String(), nullable=True),
    sa.Column('eda_word_freq_url', sa.String(), nullable=True),
    sa.Column('eda_bigram_url', sa.String(), nullable=True),
    sa.Column('eda_trigram_url', sa.String(), nullable=True),
    sa.Column('eda_updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('file_hash'),
    sa.UniqueConstraint('s3_key')
    )
    op.create_index(op.f('ix_file_records_id'), 'file_records', ['id'], unique=False)
    op.create_table('topic_model',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('file_id', sa.String(), nullable=False),
    sa.Column('method', sa.String(), nullable=False),
    sa.Column('topic_count', sa.Integer(), nullable=True),
    sa.Column('s3_key', sa.String(), nullable=False),
    sa.Column('s3_url', sa.String(), nullable=False),
    sa.Column('summary_json', sa.String(), nullable=True),
    sa.Column('label_keywords', sa.String(), nullable=True),
    sa.Column('label_map_json', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['file_id'], ['file_records.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('s3_key')
    )
    op.create_index(op.f('ix_topic_model_id'), 'topic_model', ['id'], unique=False)
    op.create_table('sentiment_analysis',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('topic_model_id', sa.String(), nullable=False),
    sa.Column('method', sa.String(), nullable=False),
    sa.Column('overall_positive', sa.Float(), nullable=False),
    sa.Column('overall_neutral', sa.Float(), nullable=False),
    sa.Column('overall_negative', sa.Float(), nullable=False),
    sa.Column('per_topic_json', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['topic_model_id'], ['topic_model.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_sentiment_analysis_id'), 'sentiment_analysis', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_sentiment_analysis_id'), table_name='sentiment_analysis')
    op.drop_table('sentiment_analysis')
    op.drop_index(op.f('ix_topic_model_id'), table_name='topic_model')
    op.drop_table('topic_model')
    op.drop_index(op.f('ix_file_records_id'), table_name='file_records')
    op.drop_table('file_records')
    # ### end Alembic commands ###
