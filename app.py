import asyncio,time,hmac,logging,uuid
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Header,Depends,Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field,model_validator
from prometheus_client import Counter,Histogram,generate_latest,CONTENT_TYPE_LATEST
from starlette.responses import Response
import jwt
from rag import config
from rag.index import ensure,embedder
from rag.cache import get_exact,put,close
from rag.pii import mask_pii
from rag.answer import generate_and_verify, llm
from rag import conversations
from rag.limits import enforce_limits
from rag.query_understanding import classify_query, assess_corpus_compatibility
from rag.lexical import domain_counts
from celery_app import celery

log=logging.getLogger("rag")
REQ=Counter("rag_requests_total","Requests",["result"])
LAT=Histogram("rag_latency_seconds","End-to-end latency")
SEM=asyncio.Semaphore(config.MAX_GENERATIONS)

@asynccontextmanager
async def lifespan(app):
    if len(config.API_KEY)<32: raise RuntimeError("Set a 32+ character API_KEY")
    if len(config.JWT_SECRET)<32: raise RuntimeError("Set a 32+ character JWT_SECRET")
    ensure(); await conversations.init_db()
    await asyncio.to_thread(embedder().encode,["warmup"],normalize_embeddings=True)
    yield
    await conversations.close_db(); await close()

app=FastAPI(title="Private PDF Agentic RAG",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=config.CORS,allow_credentials=False,
                   allow_methods=["GET","POST"],allow_headers=["Authorization","Content-Type"])

async def current_user(authorization:str=Header(default="")):
    token=authorization.removeprefix("Bearer ")
    if hmac.compare_digest(authorization,"Bearer "+config.API_KEY):
        return "service-admin"
    try:
        claims=jwt.decode(token,config.JWT_SECRET,algorithms=[config.JWT_ALGORITHM],options={"require":["sub","exp"]})
        user_id=str(claims["sub"])
        if not user_id or len(user_id)>200: raise ValueError("bad subject")
        return user_id
    except Exception:
        raise HTTPException(401,"A valid user access token is required")

async def current_tenant(authorization:str=Header(default="")):
    if hmac.compare_digest(authorization,"Bearer "+config.API_KEY): return "default"
    token=authorization.removeprefix("Bearer ")
    try:
        claims=jwt.decode(token,config.JWT_SECRET,algorithms=[config.JWT_ALGORITHM],options={"require":["sub","exp"]})
        tenant=str(claims.get("tenant_id") or claims["sub"])
        if not tenant or len(tenant)>128: raise ValueError("bad tenant")
        return tenant
    except Exception: raise HTTPException(401,"A valid tenant-scoped user token is required")

class Query(BaseModel):
    question:str=Field(min_length=2,max_length=4000)
class SessionCreate(BaseModel):
    title:str=Field(default="New conversation",max_length=80)
class MetadataFilters(BaseModel):
    document_ids:list[str]=Field(default_factory=list,max_length=100)
    filenames:list[str]=Field(default_factory=list,max_length=100)
    pages:list[int]=Field(default_factory=list,max_length=200)
    page_from:int|None=Field(default=None,ge=1)
    page_to:int|None=Field(default=None,ge=1)
    extraction_methods:list[str]=Field(default_factory=list,max_length=5)
    chunk_types:list[str]=Field(default_factory=list,max_length=10)
    document_types:list[str]=Field(default_factory=list,max_length=30)
    domains:list[str]=Field(default_factory=list,max_length=20)
    tags:list[str]=Field(default_factory=list,max_length=30)
    @model_validator(mode="after")
    def validate_values(self):
        allowed_extract={"native","page_ocr"}; allowed_chunks={"text_or_table"}
        allowed_domains={"medical","finance","legal","hr","technology","insurance","education","general"}
        if any(x not in allowed_extract for x in self.extraction_methods): raise ValueError("Unsupported extraction method")
        if any(x not in allowed_chunks for x in self.chunk_types): raise ValueError("Unsupported chunk type")
        if any(x not in allowed_domains for x in self.domains): raise ValueError("Unsupported domain filter")
        if self.page_from and self.page_to and self.page_from>self.page_to: raise ValueError("page_from must be <= page_to")
        return self

class SessionQuery(BaseModel):
    question:str=Field(min_length=2,max_length=4000)
    filters:MetadataFilters=Field(default_factory=MetadataFilters)

class IngestMetadata(BaseModel):
    tenant_id:str=Field(default="default",min_length=1,max_length=128)
    document_type:str=Field(default="general",max_length=80)
    domain:str=Field(default="general",max_length=40)
    tags:list[str]=Field(default_factory=list,max_length=30)
    source:str=Field(default="upload",max_length=80)
    @model_validator(mode="after")
    def validate_domain(self):
        if self.domain not in {"medical","finance","legal","hr","technology","insurance","education","general"}:
            raise ValueError("Unsupported document domain")
        return self

@app.get("/health")
async def health():return {"status":"ok"}
@app.get("/metrics")
async def metrics():return Response(generate_latest(),media_type=CONTENT_TYPE_LATEST)

@app.post("/sessions")
async def create_session(user_id:str=Depends(current_user)):
    sid=uuid.uuid4(); expiry=datetime.now(timezone.utc)+timedelta(days=config.SESSION_TTL_DAYS)
    await conversations.create_session(user_id,sid,expiry)
    return {"session_id":str(sid),"expires_at":expiry.isoformat()}

@app.get("/sessions")
async def list_sessions(user_id:str=Depends(current_user)):
    return {"sessions":await conversations.list_sessions(user_id)}

@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id:uuid.UUID,user_id:str=Depends(current_user)):
    if not await conversations.get_session(user_id,session_id): raise HTTPException(404,"Session not found")
    return {"messages":await conversations.history(user_id,session_id,limit=100)}

async def compact_history(user_id, sid, session_row):
    all_msgs=await conversations.history(user_id,sid,limit=500)
    summary=session_row["summary"] or ""
    summary_count=session_row["summary_count"] or 0
    keep=config.HISTORY_RECENT_MESSAGES
    cutoff=max(0,len(all_msgs)-keep)
    if cutoff>summary_count and len(all_msgs)>=config.SUMMARY_TRIGGER_MESSAGES:
        old=all_msgs[summary_count:cutoff]
        transcript="\n".join(f"{m['role']}: {m['content']}" for m in old)
        prompt=("Update this concise factual conversation memory. Preserve user goals, constraints, "
                "entities, decisions and unresolved questions. Drop greetings and repetition. "
                "Do not invent facts. Keep under 700 words.\n\nExisting memory:\n"+summary+
                "\n\nNew messages to incorporate:\n"+transcript)
        summary=await llm([{"role":"system","content":"You maintain a compact conversation memory."},{"role":"user","content":prompt}],max_tokens=650)
        await conversations.update_summary(user_id,sid,summary,cutoff)
    recent=all_msgs[max(cutoff,summary_count):]
    # Hard character budget; retain newest messages first.
    budget=config.HISTORY_MAX_CHARS; selected=[]
    for m in reversed(recent):
        item={"role":m["role"],"content":m["content"]}
        cost=len(item["content"])
        if cost>budget: item["content"]=item["content"][-budget:]; cost=budget
        if cost<=0: break
        selected.append(item); budget-=cost
        if budget<=0: break
    return summary,list(reversed(selected))

def query_gate(question: str, tenant_id: str, filters: MetadataFilters | None = None, context: str = ""):
    # Use prior user context to classify elliptical follow-ups such as "what about symptoms?".
    analysis = classify_query((context[-1800:] + "\nCurrent question: " + question) if context else question)
    # Caller-specified domain filters are authoritative constraints, but never override tenant scope.
    counts = domain_counts(tenant_id)
    compatibility = assess_corpus_compatibility(analysis, counts, config.QUERY_DOMAIN_MIN_CONFIDENCE)
    if filters and filters.domains:
        selected = set(filters.domains)
        if analysis["domain"] != "unknown" and analysis["domain"] not in selected:
            compatibility = {"status": "domain_mismatch", "reason": "query_conflicts_with_requested_domain_filter", "available_domains": counts}
    return analysis, compatibility

def clarification_response(analysis: dict, compatibility: dict):
    return {"answer": "Your question appears to span multiple domains. Could you clarify which domain or document collection you want me to use?",
            "sources": [], "verified": False, "verification": "query_clarification_required",
            "query_analysis": analysis, "domain_check": compatibility, "retrieved_chunks": 0}

def mismatch_response(analysis: dict, compatibility: dict):
    requested = analysis.get("domain", "unknown")
    available = sorted(k for k,v in compatibility.get("available_domains", {}).items() if v > 0)
    return {"answer": (f"Your question appears to concern {requested} information, but this knowledge base does not contain a matching domain. "
            f"Available indexed domains: {', '.join(available) if available else 'not yet labelled'}. "
            "Please select a matching knowledge base or connect relevant documents."),
            "sources": [], "verified": False, "verification": "domain_mismatch",
            "query_analysis": analysis, "domain_check": compatibility, "retrieved_chunks": 0}

@app.post("/sessions/{session_id}/chat")
async def session_chat(session_id:uuid.UUID,q:SessionQuery,user_id:str=Depends(current_user),tenant_id:str=Depends(current_tenant)):
    start=time.perf_counter()
    session=await conversations.get_session(user_id,session_id)
    if not session: raise HTTPException(404,"Session not found or expired")
    await enforce_limits(user_id,str(session_id))
    safe_question,pii_count=mask_pii(q.question)
    # Context-aware classification handles follow-ups without letting old assistant claims become evidence.
    summary,history=await compact_history(user_id,session_id,session)
    prior_user_context="\n".join(m["content"] for m in history if m["role"] == "user")
    analysis, compatibility = query_gate(safe_question, tenant_id, q.filters, context=(summary+"\n"+prior_user_context))
    # Persist the user turn even when routing stops early, so the session remains coherent.
    await conversations.add_message(user_id,session_id,"user",safe_question)
    effective_filters=q.filters.model_dump(exclude_none=True)
    if not effective_filters.get("domains") and analysis["domain"] in compatibility.get("available_domains", {}):
        effective_filters["domains"]=[analysis["domain"]]
    if compatibility["status"] == "domain_mismatch":
        result=mismatch_response(analysis, compatibility)
    elif compatibility["status"] == "clarify":
        result=clarification_response(analysis, compatibility)
    else:
        async with SEM:
            result=await generate_and_verify(safe_question,history=history,summary=summary,tenant_id=tenant_id,filters=effective_filters)
        result["query_analysis"] = analysis
        result["domain_check"] = compatibility
    await conversations.add_message(user_id,session_id,"assistant",result["answer"])
    result.update({"session_id":str(session_id),"pii_entities_masked":pii_count,
                   "latency_ms":round((time.perf_counter()-start)*1000,1),"cache":"bypassed_session_context"})
    REQ.labels(result="session_success").inc(); LAT.observe(time.perf_counter()-start)
    return result

# Stateless compatibility endpoint; session endpoint is recommended for real users.
@app.post("/chat")
async def chat(q:Query,user_id:str=Depends(current_user),tenant_id:str=Depends(current_tenant),filters:MetadataFilters=Depends()):
    start=time.perf_counter(); await enforce_limits(user_id,"stateless")
    try:
        safe_question,pii_count=mask_pii(q.question)
        analysis, compatibility = query_gate(safe_question, tenant_id, filters)
        effective_filters=filters.model_dump(exclude_none=True)
        if not effective_filters.get("domains") and analysis["domain"] in compatibility.get("available_domains", {}):
            effective_filters["domains"]=[analysis["domain"]]
        if compatibility["status"] in {"domain_mismatch", "clarify"}:
            result=(mismatch_response(analysis, compatibility) if compatibility["status"] == "domain_mismatch"
                    else clarification_response(analysis, compatibility))
            result.update({"cache":"bypassed_domain_mismatch","pii_entities_masked":pii_count,
                           "latency_ms":round((time.perf_counter()-start)*1000,1)})
            REQ.labels(result="domain_mismatch").inc(); LAT.observe(time.perf_counter()-start)
            return result
        cached=await get_exact(safe_question,tenant_id,effective_filters)
        if cached:
            cached["query_analysis"]=analysis; cached["domain_check"]=compatibility
            cached["cache"]="hit"; cached["latency_ms"]=round((time.perf_counter()-start)*1000,1)
            REQ.labels(result="cache_hit").inc();LAT.observe(time.perf_counter()-start);return cached
        async with SEM: result=await generate_and_verify(safe_question,tenant_id=tenant_id,filters=effective_filters)
        result.update({"cache":"miss","pii_entities_masked":pii_count,"latency_ms":round((time.perf_counter()-start)*1000,1),
                       "query_analysis":analysis,"domain_check":compatibility})
        if result.get("verified"): await put(safe_question,result,tenant_id,effective_filters)
        REQ.labels(result="success" if result.get("verified") else "unverified").inc();LAT.observe(time.perf_counter()-start)
        return result
    except HTTPException: raise
    except Exception:
        REQ.labels(result="error").inc();log.exception("RAG processing failed; no document/question text logged")
        raise HTTPException(502,"RAG processing failed")

@app.post("/admin/ingest/{filename}")
async def enqueue_ingestion(filename:str,metadata:IngestMetadata=Body(default=IngestMetadata()),user_id:str=Depends(current_user)):
    if user_id!="service-admin": raise HTTPException(403,"Admin service credential required")
    if "/" in filename or "\\" in filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(400,"Provide a PDF filename located in the configured PDF directory")
    task=celery.send_task("rag.tasks.ingest_pdf",args=[filename,metadata.tenant_id,metadata.model_dump(exclude={"tenant_id"})],queue="ingestion")
    return {"task_id":task.id,"status":"queued"}
