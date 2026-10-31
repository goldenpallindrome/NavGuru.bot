from functools import lru_cache
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient,models
from . import config

@lru_cache
def embedder(): return SentenceTransformer(config.EMBEDDING_MODEL,device=config.EMBEDDING_DEVICE)
@lru_cache
def client(): return QdrantClient(url=config.QDRANT_URL,timeout=30)
def ensure():
    c=client()
    if not c.collection_exists(config.COLLECTION):
        dim=embedder().get_sentence_embedding_dimension()
        c.create_collection(config.COLLECTION,vectors_config=models.VectorParams(
          size=dim,distance=models.Distance.COSINE),
          hnsw_config=models.HnswConfigDiff(m=16,ef_construct=128))
    # Idempotently ensure payload indexes even for collections created by an older release.
    for field,typ in [("document_id",models.PayloadSchemaType.KEYWORD),
                      ("filename",models.PayloadSchemaType.KEYWORD),
                      ("page",models.PayloadSchemaType.INTEGER),
                      ("tenant_id",models.PayloadSchemaType.KEYWORD),
                      ("document_type",models.PayloadSchemaType.KEYWORD),
                      ("domain",models.PayloadSchemaType.KEYWORD),
                      ("tags",models.PayloadSchemaType.KEYWORD),
                      ("extraction",models.PayloadSchemaType.KEYWORD),
                      ("chunk_type",models.PayloadSchemaType.KEYWORD)]:
        c.create_payload_index(config.COLLECTION,field,field_schema=typ)
def vectorize(texts):
    return embedder().encode(texts,normalize_embeddings=True,batch_size=32,
                             show_progress_bar=False).tolist()
def index_chunks(rows):
    ensure(); vecs=vectorize([r["text"] for r in rows])
    pts=[models.PointStruct(id=r["id"],vector=v,payload=r) for r,v in zip(rows,vecs)]
    for i in range(0,len(pts),128):
        client().upsert(collection_name=config.COLLECTION,points=pts[i:i+128],wait=True)
def delete_document(doc_id):
    if client().collection_exists(config.COLLECTION):
        client().delete(collection_name=config.COLLECTION,points_selector=models.FilterSelector(
          filter=models.Filter(must=[models.FieldCondition(key="document_id",
          match=models.MatchValue(value=doc_id))])),wait=True)
