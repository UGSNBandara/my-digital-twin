"""
app.py
======
FastAPI entrypoint.
- Loads .env for local development
- CORS configured for portfolio domain only
- Rate limiting via slowapi (10 req/min per IP)
- X-Portfolio-Key header check on /chat (set server-side by Vercel proxy)
- FAISS index built at startup
- Session cleanup background task started at startup
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(override=True)  # no-op on HuggingFace (env vars injected by HF Secrets)

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

import agent
import session as session_store
from security import ALLOWED_ORIGINS, limiter, verify_portfolio_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building FAISS knowledge index...")
    agent.build_index()
    logger.info("Starting session cleanup task...")
    cleanup_task = asyncio.create_task(session_store.cleanup_loop())
    yield
    cleanup_task.cancel()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Sulitha's Portfolio Agent",
    description = "AI agent that represents Sulitha Nulaksha Bandara.",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = None,
    redoc_url   = None,
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = False,
    allow_methods     = ["POST", "GET"],
    allow_headers     = ["Content-Type", "X-Portfolio-Key"],
)


# ── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code = 429,
        content     = {"detail": "Too many messages — please wait a moment and try again."},
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str
    session_id: str | None = None   # None on first message; frontend stores and resends


class ChatResponse(BaseModel):
    reply:      str
    session_id: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, body: ChatRequest):
    """
    Security order:
      1. X-Portfolio-Key header  — rejects anything not coming from the Vercel proxy
      2. Rate limit              — 10 requests/min per IP
      3. Session message cap     — checked inside agent.chat
    """
    verify_portfolio_key(request)

    session_id = body.session_id or str(uuid.uuid4())

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(message) > 1000:
        raise HTTPException(status_code=400, detail="Message too long (max 1000 chars).")

    reply = await agent.chat(message, session_id)

    return ChatResponse(reply=reply, session_id=session_id)
