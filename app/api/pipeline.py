import math
from app.core.database import async_session
import json
from fastapi import APIRouter, Depends, Query, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.core.database import get_db
from app.core.eda.config import EDAConfig
from app.core.eda.eda_analyzer import DefaultEDAAnalyzer
from app.core.file_handler.codec import CsvCodec
from app.core.file_handler.compression import GzipCompression
from app.core.file_handler.storage import S3Storage
from app.core.lemmatization.config import LemmatizationConfig
from app.core.lemmatization.lemmatizer import DefaultLemmatizer
from app.core.normalization.normalizer import DefaultTextNormalizer
from app.core.sentiment.analyzer import analyzer_for
from app.core.sentiment.config import SentimentConfig
from app.core.special_char_removal.cleaner import DefaultSpecialCleaner
from app.core.special_char_removal.config import SpecialCleanConfig
from app.core.stopword_removal.config import StopwordConfig
from app.core.stopword_removal.removal import DefaultStopwordRemover
from app.core.tokenization.config import TokenizationConfig
from app.core.tokenization.tokenizer import DefaultTokenizer
from app.core.topic_labeling.config import TopicLabelConfig
from app.core.topic_labeling.labelers import DefaultHeuristicLabeler
from app.core.topic_modeling.config import TopicEstimationConfig, TopicModelConfig
from app.core.topic_modeling.gensim_lda import GensimLDAModeler
from app.models.db.file_record import FileRecord
from app.models.db.topic_model import TopicModel
from app.models.db.sentiment_analysis import SentimentAnalysis
from app.schemas.pipeline import FullPipelineRequest
from app.schemas.sentiment import SentimentTopicBreakdown
from app.services.eda_service import EDAService
from app.services.file_service import FileService
from app.services.lemmatization_service import LemmatizationService
from app.services.normalization_service import NormalizationService
from app.services.sentiment_service import SentimentService
from app.services.special_clean_service import SpecialCleanService
from app.services.stopword_service import StopwordService
from app.services.tokenization_service import TokenizationService

from app.services.topic_labeling_service import TopicLabelingService
from app.services.topic_modeling_service import TopicModelingService
from app.utils.response_builder import success_response
from app.utils.exceptions import NotFoundError, ServerError
from app.messages.pipeline_messages import (
    FILE_NOT_FOUND,
    SAMPLE_FILE_NOT_FOUND,
    FULL_PIPELINE_SUCCESS,
    SAMPLE_DATA_URL_SUCCESS,
)


router = APIRouter(prefix="/api/pipeline", tags=["Full Pipeline"])
logger = logging.getLogger(__name__)


def _files() -> FileService:
    return FileService(
        storage=S3Storage(),
        codec=CsvCodec(index=False),
        compression=GzipCompression(),
    )


def safe_pct(numerator: int, denominator: int) -> float:
    if not denominator:
        return 0.0
    v = (numerator / denominator) * 100.0
    if not math.isfinite(v):
        return 0.0
    return round(v, 1)


async def run_sentiment_pipeline(file_id: str):
    async with async_session() as db:
        try:
            logger.info(
                f"Running sentiment pipeline background task with file_id: {file_id}"
            )
            # Check if file record exists
            record = (
                await db.execute(select(FileRecord).filter_by(id=file_id))
            ).scalar_one_or_none()

            if not record:
                raise NotFoundError(code="FILE_NOT_FOUND", message=FILE_NOT_FOUND)

            # Check if topic model exists
            existing_topic_model = (
                (
                    await db.execute(
                        select(TopicModel)
                        .filter_by(file_id=file_id, method="LDA")
                        .order_by(TopicModel.label_updated_at.desc())
                    )
                )
                .scalars()
                .first()
            )

            # If topic model exists, check for existing sentiment analysis
            if existing_topic_model:
                existing_sentiment = (
                    (
                        await db.execute(
                            select(SentimentAnalysis)
                            .filter_by(
                                topic_model_id=existing_topic_model.id, method="vader"
                            )
                            .order_by(SentimentAnalysis.updated_at.desc())
                        )
                    )
                    .scalars()
                    .first()
                )

                if (
                    existing_sentiment
                    and record.lemmatized_updated_at
                    and existing_sentiment.updated_at >= record.lemmatized_updated_at
                ):
                    logger.info("⏩ Skipping pipeline — sentiment already up-to-date.")
                    per_topic = [
                        SentimentTopicBreakdown(**t)
                        for t in json.loads(existing_sentiment.per_topic_json or "[]")
                    ]
                    return success_response(
                        message="Sentiment already computed. Reusing existing result.",
                        data={
                            "overall": {
                                "positive": existing_sentiment.overall_positive,
                                "neutral": existing_sentiment.overall_neutral,
                                "negative": existing_sentiment.overall_negative,
                            },
                            "per_topic": [t.model_dump() for t in per_topic],
                        },
                    )

            # Otherwise, Process normal flow:
            # Data Cleaning -> EDA -> Pick K Topic & Topic Modeling -> Topic Labeling -> Sentiment Analysis
            files = FileService(
                storage=S3Storage(),
                codec=CsvCodec(index=False),
                compression=GzipCompression(),
            )

            # Step A.1: Normalization
            normalizer = DefaultTextNormalizer()
            norm_service = NormalizationService(files, normalizer)

            norm_result = await norm_service.ensure_normalized(db=db, record=record)
            df = norm_result.df  # ['review', 'normalized_review']

            # Step A.2: Special characters
            cleaner = DefaultSpecialCleaner(
                SpecialCleanConfig(
                    remove_special=True, remove_numbers=True, remove_emoji=True
                )
            )
            special_service = SpecialCleanService(files, cleaner)

            special_result = await special_service.ensure_special_cleaned(
                db=db,
                record=record,
                df_norm=df,  # pass in-memory df to avoid S3 read
                override_flags=None,  # or {"remove_special": True, ...}
            )
            df = special_result.df  # ['normalized_review', 'special_cleaned']

            # Step A.3: Tokenization
            tokenizer = DefaultTokenizer(TokenizationConfig(method="wordpunct"))
            tok_service = TokenizationService(files, tokenizer)

            tok_result = await tok_service.ensure_tokenized(
                db=db,
                record=record,
                df_clean=df,
                override_config=None,
            )
            df = tok_result.df  # columns: ['special_cleaned', 'tokens']

            # Step A.4: Stopword Removal
            remover = DefaultStopwordRemover(
                StopwordConfig(
                    language="english",
                    custom_stopwords=set(),  # e.g., {"hotel","room"} for domain noise
                    exclude_stopwords=set(),  # keep these even if they’re stopwords
                    lowercase=True,
                    preserve_negations=True,
                )
            )
            stop_service = StopwordService(files, remover)

            stop_result = await stop_service.ensure_stopwords_removed(
                db=db,
                record=record,
                df_tokens=df,
            )
            df = stop_result.df  # columns: ['tokens', 'stopword_removed']

            # Step A.5: Lemmatization
            lemmatizer = DefaultLemmatizer(
                LemmatizationConfig(use_pos_tagging=True, lowercase=False)
            )
            lem_service = LemmatizationService(files, lemmatizer)

            lem_result = await lem_service.ensure_lemmatized(
                db=db, record=record, df_stop=stop_result.df
            )
            df = lem_result.df  # ['stopword_removed', 'lemmatized_tokens']

            # Step B.1: Exploratory Data Analysis
            eda_analyzer = DefaultEDAAnalyzer(
                EDAConfig(
                    text_column="lemmatized_tokens", top_words=100, ngram_top_k=20
                )
            )
            eda_service = EDAService(analyzer=eda_analyzer)

            await eda_service.ensure_eda(
                db=db,
                record=record,
                df_lemm=lem_result.df,  # <- pass in-memory DF from the lemmatization step
            )

            # Step C.1: Pick K Topic & Topic modeling
            tm_cfg = TopicModelConfig(backend="gensim", passes=10, topn_words=10)
            est_cfg = TopicEstimationConfig(method="coherence", min_k=3, max_k=5)

            modeler = GensimLDAModeler(tm_cfg)
            estimator = modeler

            tm_service = TopicModelingService(
                files=files, modeler=modeler, estimator=estimator
            )
            tm_res = await tm_service.ensure_topics(
                db=db,
                record=record,
                df_lemm=lem_result.df,
                cfg=tm_cfg,
                est=est_cfg,
                force_k=3,  # or force_k=your_fixed_number
            )
            df = tm_res.df  # has ['topic_id','topic_label']

            # Step D.1: Assign Default Topic Labels
            label_svc = TopicLabelingService(
                files=files,
                labeler=DefaultHeuristicLabeler(
                    TopicLabelConfig(strategy="default", num_keywords=3)
                ),
            )

            lda_entry = (
                await db.execute(select(TopicModel).filter_by(id=tm_res.topic_model_id))
            ).scalar_one()

            label_res = await label_svc.ensure_labeled(
                db=db, topic_model=lda_entry, df_topics=df
            )
            df = label_res.df  # contains 'topic_label'

            # Step E.1: Sentiment Analysis
            sent_svc = SentimentService(
                files=files,
                analyzer=analyzer_for("vader"),
                cfg=SentimentConfig(method="vader"),
            )
            sent_res = await sent_svc.ensure_sentiment(
                db=db, topic_model=lda_entry, df_labeled=df
            )

            overall = sent_res.overall
            per_topic = [t.model_dump() for t in sent_res.per_topic]

            logger.info(f"Background sentiment task completed: {file_id}")

            return success_response(
                message=FULL_PIPELINE_SUCCESS,
                data={"status": "done", "overall": overall, "per_topic": per_topic},
            )

        except Exception as e:
            logger.exception(f"❌ Full pipeline failed in background: {e}")


@router.post("/sentiment-analysis")
async def run_full_pipeline(
    req: FullPipelineRequest,
    background_tasks: BackgroundTasks,
):
    try:
        background_tasks.add_task(run_sentiment_pipeline, req.file_id)

        return {
            "message": "✅ Pipeline started in background.",
            "data": {
                "file_id": req.file_id,
                "status": "processing",
            },
        }

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"❌ Full pipeline failed: {e}")
        raise ServerError(
            code="FULL_PIPELINE_FAILED",
            message="Full pipeline sentiment analysis failed.",
        )


@router.get("/result")
async def get_result(file_id: str = Query(...), db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(FileRecord).filter_by(id=file_id))
        record = result.scalar_one_or_none()

        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message=SAMPLE_FILE_NOT_FOUND)

        topic_model = (
            (
                await db.execute(
                    select(TopicModel)
                    .filter_by(file_id=file_id, method="LDA")
                    .order_by(TopicModel.label_updated_at.desc())
                )
            )
            .scalars()
            .first()
        )

        sentiment = None
        if topic_model:
            sentiment = (
                (
                    await db.execute(
                        select(SentimentAnalysis)
                        .filter_by(topic_model_id=topic_model.id, method="vader")
                        .order_by(SentimentAnalysis.updated_at.desc())
                    )
                )
                .scalars()
                .first()
            )

        # If neither is present, it's still processing
        if not topic_model or not sentiment:
            return success_response(
                message="Sentiment analysis still processing.",
                data={
                    "file_id": file_id,
                    "status": "processing",
                },
            )

        per_topic = [
            SentimentTopicBreakdown(**t)
            for t in json.loads(sentiment.per_topic_json or "[]")
        ]

        return success_response(
            message="Sentiment analysis result loaded successfully.",
            data={
                "file_id": file_id,
                "status": "done",
                "overall": {
                    "positive": sentiment.overall_positive,
                    "neutral": sentiment.overall_neutral,
                    "negative": sentiment.overall_negative,
                },
                "per_topic": [t.model_dump() for t in per_topic],
            },
        )

    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.exception(f"❌ Failed to load result: {e}")
        raise ServerError(
            code="RESULT_LOAD_FAILED", message="Failed to load sentiment result."
        )


@router.get("/sample-data-url")
async def get_sample_data_url(
    file_id: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    try:
        # Look up the record so we use the real stored key
        result = await db.execute(select(FileRecord).filter_by(id=file_id))
        record = result.scalars().first()
        if not record:
            raise NotFoundError(code="FILE_NOT_FOUND", message="File not found.")

        if not record.s3_key:
            raise NotFoundError(
                code="SOURCE_S3_KEY_MISSING",
                message="Source file key is missing for this record.",
            )

        files = _files()
        s3_url = await files.presigned_url(record.s3_key, expires_in=6000)

        return success_response(
            message=SAMPLE_DATA_URL_SUCCESS,
            data={"s3_url": s3_url},
        )

    except NotFoundError:
        raise
    except Exception as e:
        logger.exception(f"❌ Failed to generate sample data URL: {e}")
        raise ServerError(
            code="SAMPLE_DATA_URL_FAILED",
            message="Failed to generate sample data URL.",
        )
