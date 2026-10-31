"""Redis fixed-window user/session limits. Use atomic INCR + EXPIRE."""
from .cache import rc
from . import config
from fastapi import HTTPException

async def enforce_limits(user_id: str, session_id: str):
    r = rc()
    checks = [
      (f"rag:rl:user:{user_id}", config.USER_RPM, 60),
      (f"rag:rl:session:{session_id}", config.SESSION_RPM, 60),
      ("rag:rl:global", config.GLOBAL_RPM, 60),
    ]
    for key, maximum, window in checks:
        n = await r.incr(key)
        if n == 1: await r.expire(key, window)
        if n > maximum:
            ttl = await r.ttl(key)
            raise HTTPException(429, detail={"error":"rate_limit_exceeded","retry_after_seconds":max(1,ttl)})
