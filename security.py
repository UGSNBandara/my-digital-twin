"""
security.py
===========
Security layers:
  1. CORS              — portfolio domain only
  2. X-Portfolio-Key   — shared secret between Vercel proxy and this backend
                         (secret lives in Vercel server env, never in browser JS)
  3. Rate limiting     — slowapi, 10 req/min per IP
"""

import os
import logging
from fastapi import Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# ── Secrets ───────────────────────────────────────────────────────────────────
PORTFOLIO_SECRET_KEY = os.environ.get("PORTFOLIO_SECRET_KEY", "")

# ── Allowed origins ───────────────────────────────────────────────────────────
# Add your Vercel domain here. localhost entries are for local dev only.
ALLOWED_ORIGINS = [
    "https://sulitha-nulaksha-portfolio.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
]

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])


# ── Origin token check ────────────────────────────────────────────────────────

def verify_portfolio_key(request: Request) -> None:
    """
    Reject requests that don't carry the correct X-Portfolio-Key header.
    This header is set server-side by the Vercel API route proxy — it never
    reaches the browser, so it cannot be scraped from frontend JS.
    Skipped with a warning if PORTFOLIO_SECRET_KEY is not set (useful for
    local dev without the full proxy setup).
    """
    if not PORTFOLIO_SECRET_KEY:
        logger.warning("PORTFOLIO_SECRET_KEY not set — skipping key check (dev mode).")
        return

    key = request.headers.get("X-Portfolio-Key", "")
    if key != PORTFOLIO_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
