from pathlib import Path
from celery_app import celery
from . import config
from .ingest import process
from .index import index_chunks, delete_document
from .lexical import replace_document

@celery.task(name="rag.tasks.ingest_pdf", bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries":3})
def ingest_pdf(self, filename: str, tenant_id: str="default", metadata: dict|None=None):
    root = Path(config.PDF_DIR).resolve()
    path = (root / filename).resolve()
    if root not in path.parents or path.suffix.lower() != ".pdf" or not path.is_file():
        raise ValueError("Invalid PDF path")
    doc_id, rows = process(path,tenant_id=tenant_id,metadata=metadata or {})
    delete_document(doc_id)
    index_chunks(rows)
    replace_document(doc_id, rows)
    return {"document_id":doc_id,"filename":path.name,"chunks_indexed":len(rows)}
