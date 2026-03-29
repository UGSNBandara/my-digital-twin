"""
session.py
==========
In-memory session store with automatic TTL cleanup.
Sessions are dicts keyed by session_id (UUID from frontend).
A background task purges sessions inactive for SESSION_TTL_MINUTES.
"""

import time
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SESSION_TTL_MINUTES   = 30   # delete session after this many minutes of inactivity
CLEANUP_INTERVAL_SECS = 600  # run cleanup every 10 minutes
MAX_HISTORY_TURNS     = 10   # keep last N user+assistant pairs in memory
SESSION_MSG_LIMIT     = 20   # hard cap — after this, agent redirects to email

# ── Store ─────────────────────────────────────────────────────────────────────
_sessions: dict[str, dict] = {}


# ── Public API ────────────────────────────────────────────────────────────────

def get_or_create(session_id: str) -> dict:
    """Return existing session or create a fresh one."""
    if session_id not in _sessions:
        _sessions[session_id] = {
            "history":       [],   # list of {"role": ..., "content": ...}
            "message_count": 0,
            "last_active":   time.time(),
        }
    return _sessions[session_id]


def append(session_id: str, role: str, content: str) -> None:
    """Append a message to session history and update last_active."""
    session = get_or_create(session_id)
    session["history"].append({"role": role, "content": content})
    session["message_count"] += 1
    session["last_active"] = time.time()

    # Trim to MAX_HISTORY_TURNS pairs (keep most recent)
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(session["history"]) > max_msgs:
        session["history"] = session["history"][-max_msgs:]


def get_history(session_id: str) -> list[dict]:
    """Return conversation history for a session."""
    return _sessions.get(session_id, {}).get("history", [])


def get_message_count(session_id: str) -> int:
    """Return how many messages have been sent in this session."""
    return _sessions.get(session_id, {}).get("message_count", 0)


def is_over_limit(session_id: str) -> bool:
    return get_message_count(session_id) >= SESSION_MSG_LIMIT


# ── Background Cleanup ────────────────────────────────────────────────────────

async def cleanup_loop() -> None:
    """Async background task — deletes stale sessions every CLEANUP_INTERVAL_SECS."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECS)
        _purge_stale()


def _purge_stale() -> None:
    cutoff = time.time() - SESSION_TTL_MINUTES * 60
    stale  = [sid for sid, s in _sessions.items() if s["last_active"] < cutoff]
    for sid in stale:
        del _sessions[sid]
    if stale:
        logger.info(f"Session cleanup: removed {len(stale)} stale session(s).")
