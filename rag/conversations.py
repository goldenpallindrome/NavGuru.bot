"""PostgreSQL-backed user-owned sessions and conversation history."""
import asyncpg
from . import config
_pool = None

async def pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(config.DATABASE_URL, min_size=1, max_size=config.DB_POOL_MAX)
    return _pool

async def init_db():
    p = await pool()
    async with p.acquire() as c:
        await c.execute("""
        CREATE TABLE IF NOT EXISTS rag_sessions (
          id UUID PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL DEFAULT 'New conversation',
          summary TEXT NOT NULL DEFAULT '', summary_count INTEGER NOT NULL DEFAULT 0, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), expires_at TIMESTAMPTZ NOT NULL
        );
        ALTER TABLE rag_sessions ADD COLUMN IF NOT EXISTS summary_count INTEGER NOT NULL DEFAULT 0;
        CREATE INDEX IF NOT EXISTS rag_sessions_user_updated ON rag_sessions(user_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS rag_messages (
          id BIGSERIAL PRIMARY KEY, session_id UUID NOT NULL REFERENCES rag_sessions(id) ON DELETE CASCADE,
          user_id TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant')),
          content TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS rag_messages_session_id ON rag_messages(session_id, id);
        """)

async def close_db():
    global _pool
    if _pool:
        await _pool.close(); _pool = None

async def create_session(user_id, session_id, expires_at):
    p = await pool()
    async with p.acquire() as c:
        await c.execute("INSERT INTO rag_sessions(id,user_id,expires_at) VALUES($1,$2,$3)", session_id, user_id, expires_at)

async def list_sessions(user_id, limit=50):
    p = await pool()
    async with p.acquire() as c:
        rows = await c.fetch("SELECT id,title,created_at,updated_at,expires_at FROM rag_sessions WHERE user_id=$1 AND expires_at>now() ORDER BY updated_at DESC LIMIT $2", user_id, limit)
    return [dict(r) for r in rows]

async def get_session(user_id, session_id):
    p = await pool()
    async with p.acquire() as c:
        return await c.fetchrow("SELECT id,title,summary,summary_count FROM rag_sessions WHERE id=$1 AND user_id=$2 AND expires_at>now()", session_id, user_id)

async def add_message(user_id, session_id, role, content):
    p = await pool()
    async with p.acquire() as c:
        async with c.transaction():
            ok = await c.fetchval("SELECT 1 FROM rag_sessions WHERE id=$1 AND user_id=$2 AND expires_at>now() FOR UPDATE", session_id, user_id)
            if not ok: return False
            await c.execute("INSERT INTO rag_messages(session_id,user_id,role,content) VALUES($1,$2,$3,$4)", session_id,user_id,role,content)
            await c.execute("UPDATE rag_sessions SET updated_at=now(), title=CASE WHEN title='New conversation' AND $3='user' THEN left($4,80) ELSE title END WHERE id=$1 AND user_id=$2",session_id,user_id,role,content)
    return True

async def history(user_id, session_id, limit=40):
    p = await pool()
    async with p.acquire() as c:
        rows = await c.fetch("SELECT role,content,created_at FROM (SELECT role,content,created_at,id FROM rag_messages WHERE session_id=$1 AND user_id=$2 ORDER BY id DESC LIMIT $3) x ORDER BY id ASC", session_id,user_id,limit)
    return [dict(r) for r in rows]

async def update_summary(user_id, session_id, summary, count):
    p = await pool()
    async with p.acquire() as c:
        await c.execute("UPDATE rag_sessions SET summary=$3,summary_count=$4,updated_at=now() WHERE id=$1 AND user_id=$2",session_id,user_id,summary,count)

async def get_summary(user_id, session_id):
    p = await pool()
    async with p.acquire() as c:
        return await c.fetchrow("SELECT summary,summary_count FROM rag_sessions WHERE id=$1 AND user_id=$2",session_id,user_id)
