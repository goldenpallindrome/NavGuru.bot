from celery import Celery
from rag import config
celery = Celery("private_pdf_rag", broker=config.CELERY_BROKER_URL, backend=config.CELERY_RESULT_BACKEND)
celery.conf.update(
    task_serializer="json", result_serializer="json", accept_content=["json"],
    task_track_started=True, task_acks_late=True, worker_prefetch_multiplier=1,
    task_time_limit=3600, task_soft_time_limit=3300,
    task_routes={"rag.tasks.ingest_pdf": {"queue":"ingestion"}},
    broker_connection_retry_on_startup=True,
)
