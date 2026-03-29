"""
agent.py
========
Core agent logic using real OpenAI function calling:
  1. Build FAISS index from knowledge.py at startup
  2. For each message:
       a. Check session cap
       b. RAG: embed query → FAISS top-3 chunks
       c. Build prompt and call OpenAI with tools
       d. Loop: handle tool calls until finish_reason != "tool_calls"
       e. Save to session and return reply

Tools:
  - record_user_details  → notify via Pushover when a user shares their email
  - record_unknown_question → notify via Pushover when a question can't be answered
"""

import os
import json
import logging
import numpy as np
import faiss
import requests
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from knowledge import DOCUMENTS
import session as session_store

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_MODEL   = "gpt-4o-mini"
TOP_K_CHUNKS   = 3
MAX_TOKENS     = 400
TEMPERATURE    = 0.7

SESSION_LIMIT_REPLY = (
    "We've hit the limit for this chat session — I'd love to keep the conversation "
    "going directly! Feel free to email me at nulakshastudy19@gmail.com or message "
    "me on LinkedIn: linkedin.com/in/nulaksha-bandara"
)


# ── Tool functions ────────────────────────────────────────────────────────────

def _push(text: str) -> None:
    """Send a Pushover notification. Silently skips if tokens are not set."""
    token = os.getenv("PUSHOVER_TOKEN")
    user  = os.getenv("PUSHOVER_USER")
    if not token or not user:
        logger.warning("Pushover tokens not set — skipping notification.")
        return
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": token, "user": user, "message": text},
            timeout=5,
        )
    except Exception as e:
        logger.warning(f"Pushover notification failed: {e}")


def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided") -> dict:
    """Record that a visitor shared their email and wants to stay in touch."""
    _push(f"Portfolio visitor: {name} | email: {email} | notes: {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question: str) -> dict:
    """Record a question the agent could not answer."""
    _push(f"Unanswered question on portfolio: {question}")
    return {"recorded": "ok"}


# ── Tool schemas ──────────────────────────────────────────────────────────────

_record_user_details_json = {
    "name": "record_user_details",
    "description": (
        "Use this tool to record that a visitor is interested in getting in touch "
        "and has provided their email address."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The visitor's email address",
            },
            "name": {
                "type": "string",
                "description": "The visitor's name, if they provided it",
            },
            "notes": {
                "type": "string",
                "description": "Any extra context about the conversation worth recording",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

_record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": (
        "Always use this tool to record any question you could not answer "
        "because you didn't know the answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

TOOLS = [
    {"type": "function", "function": _record_user_details_json},
    {"type": "function", "function": _record_unknown_question_json},
]


# ── Agent class ───────────────────────────────────────────────────────────────

class SulithaAgent:

    def __init__(self):
        self.name   = "Sulitha Nulaksha Bandara"
        self.openai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

        logger.info("Loading sentence-transformer model...")
        self._embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        logger.info(f"Embedding {len(DOCUMENTS)} knowledge chunks...")
        self._doc_chunks = DOCUMENTS
        embeddings = self._embedder.encode(DOCUMENTS, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings)
        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    # ── RAG ───────────────────────────────────────────────────────────────────

    def _retrieve(self, query: str, k: int = TOP_K_CHUNKS) -> str:
        q_vec = self._embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        _, idxs = self._index.search(q_vec, k)
        chunks = [self._doc_chunks[i] for i in idxs[0] if i < len(self._doc_chunks)]
        return "\n\n---\n\n".join(chunks)

    # ── System prompt ─────────────────────────────────────────────────────────

    def _system_prompt(self, rag_context: str) -> str:
        return (
            f"You are acting as {self.name}. You are answering questions on "
            f"{self.name}'s personal portfolio website — questions about career, "
            f"background, projects, skills, and experience. Represent {self.name} "
            f"faithfully. Speak in first person, naturally and confidently. "
            f"Be professional and engaging, as if talking to a potential employer "
            f"or collaborator who came across the portfolio.\n\n"
            f"If you don't know the answer to a question, use your "
            f"record_unknown_question tool to record it.\n"
            f"If the visitor seems interested in getting in touch, ask for their "
            f"email and record it using record_user_details.\n\n"
            f"## Relevant profile context:\n{rag_context}\n\n"
            f"With this context, chat with the visitor, always staying in character "
            f"as {self.name}."
        )

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    def handle_tool_call(self, tool_calls) -> list[dict]:
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            logger.info(f"Tool called: {tool_name} with {arguments}")
            tool_fn = globals().get(tool_name)
            result  = tool_fn(**arguments) if tool_fn else {"error": "unknown tool"}
            results.append({
                "role":        "tool",
                "content":     json.dumps(result),
                "tool_call_id": tool_call.id,
            })
        return results

    # ── Chat ──────────────────────────────────────────────────────────────────

    async def chat(self, message: str, session_id: str) -> str:
        if session_store.is_over_limit(session_id):
            return SESSION_LIMIT_REPLY

        rag_context = self._retrieve(message)
        history     = session_store.get_history(session_id)

        messages = (
            [{"role": "system", "content": self._system_prompt(rag_context)}]
            + history
            + [{"role": "user", "content": message}]
        )

        done = False
        response = None
        try:
            while not done:
                response = await self.openai.chat.completions.create(
                    model       = OPENAI_MODEL,
                    messages    = messages,
                    tools       = TOOLS,
                    max_tokens  = MAX_TOKENS,
                    temperature = TEMPERATURE,
                )
                if response.choices[0].finish_reason == "tool_calls":
                    assistant_msg = response.choices[0].message
                    tool_results  = self.handle_tool_call(assistant_msg.tool_calls)
                    messages.append(assistant_msg)
                    messages.extend(tool_results)
                else:
                    done = True
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return (
                "Sorry, I'm having a bit of a technical hiccup right now. "
                "Feel free to email me directly at nulakshastudy19@gmail.com!"
            )

        reply = response.choices[0].message.content.strip()
        session_store.append(session_id, "user",      message)
        session_store.append(session_id, "assistant", reply)
        return reply


# ── Module-level interface (used by app.py) ───────────────────────────────────

_agent: SulithaAgent | None = None


def build_index() -> None:
    """Called once at app startup. Initialises the agent and FAISS index."""
    global _agent
    _agent = SulithaAgent()


async def chat(message: str, session_id: str) -> str:
    return await _agent.chat(message, session_id)
