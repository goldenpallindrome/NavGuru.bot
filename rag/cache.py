import json, hashlib, re
import redis.asyncio as redis
from . import config
from .index import vectorize

_client=None
def rc():
    global _client
    if _client is None:_client=redis.from_url(config.REDIS_URL,decode_responses=True)
    return _client

def normalize(q):
    return re.sub(r"\s+"," ",q.strip().lower())

def exact_key(q,tenant_id="default",filters=None):
    # Cache is isolated by tenant and effective metadata scope.
    scope=json.dumps(filters or {},sort_keys=True,separators=(",",":"))
    raw=f"{config.CACHE_VERSION}|{tenant_id}|{scope}|{normalize(q)}"
    return "rag:answer:"+hashlib.sha256(raw.encode()).hexdigest()

async def get_exact(q,tenant_id="default",filters=None):
    raw=await rc().get(exact_key(q,tenant_id,filters))
    return json.loads(raw) if raw else None

async def put(q,value,tenant_id="default",filters=None):
    # Only verified, non-personalized answers should be cached.
    await rc().set(exact_key(q,tenant_id,filters),json.dumps(value),ex=config.CACHE_TTL)

async def invalidate_all():
    # Increment CACHE_VERSION during corpus/index/prompt/model changes.
    await rc().flushdb()

async def close():
    if _client: await _client.aclose()
