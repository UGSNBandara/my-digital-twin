"""
agent.py
========
Agent design:

  ALWAYS in every prompt:
    - profile_summary.py  →  full profile, skills, contact, ~200 tokens flat cost

  Tools (real OpenAI function calling):
    - get_project_details(names)   direct lookup by name — used when LLM knows which project
    - search_projects(description) FAISS semantic search — fallback when LLM only has a description
    - record_user_details          Pushover notification when visitor shares email
    - record_unknown_question      Pushover notification when a question can't be answered

  FAISS index:
    - Built ONCE at startup over project search_text fields only (6 small strings)
    - Never rebuilt per request
    - Query embeddings called per search_projects tool call only

  Response validation:
    - After every reply, Gemini evaluates it against quality rules
    - If it fails, GPT-4o-mini reruns with the feedback to produce a fixed reply
    - Uses structured output (Pydantic Evaluation model) via Gemini OpenAI-compat endpoint
"""

import os
import re
import json
import logging
import numpy as np
import faiss
import requests
from pydantic import BaseModel
from openai import AsyncOpenAI, OpenAI

from profile_summary import PROFILE_SUMMARY
from projects_data import get_by_names, get_all_search_texts
import session as session_store

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_MODEL   = "gpt-4o-mini"
EVAL_MODEL     = "gpt-4.1-nano"
EMBED_MODEL    = "text-embedding-3-small"
MAX_TOKENS     = 500
TEMPERATURE    = 0.7
DEFAULT_TOP_N  = 3

SESSION_LIMIT_REPLY = (
    "We've hit the limit for this chat session — I'd love to keep the conversation "
    "going directly! Feel free to email me at nulakshastudy19@gmail.com or message "
    "me on LinkedIn: linkedin.com/in/nulaksha-bandara"
)

# ── Evaluator prompts ─────────────────────────────────────────────────────────

EVALUATOR_SYSTEM_PROMPT = """
You are a strict quality checker for an AI agent that represents Sulitha Nulaksha Bandara
on his personal portfolio website. Your job is to evaluate whether the agent's reply
meets all the rules below. Be precise and critical.

Rules the reply MUST follow:
1. No markdown formatting — no asterisks (*), hashes (#), underscores (_), backticks (`),
   tildes (~), or any other markdown syntax. Plain text only.
2. No emojis or special Unicode symbols.
3. Only English letters, numbers, and standard punctuation (.,!?;:'-) are allowed.
4. Concise — maximum 60 words. Default should be 1-2 sentences.
   If the topic is large, the reply must summarise briefly and offer to go deeper,
   not dump everything at once.
5. Never breaks character — must not say "As an AI", "I am a language model",
   "I don't have feelings", or anything that reveals it is not Sulitha.
6. Never fabricates information — must not claim skills, projects, awards, or facts
   that are not in the provided profile.
7. Stays on topic — only discusses Sulitha's career, projects, skills, education,
   research, and availability. Politely redirects off-topic questions.
8. Professional and friendly tone — warm but not stiff. No slang, no overly casual
   language, but also no corporate formality like "Greetings" or "How may I assist you".

Return is_acceptable as true only if ALL rules pass.
If any rule fails, set is_acceptable to false and explain exactly which rule failed
and what needs to change in the feedback field.
""".strip()


def _evaluator_user_prompt(reply: str, message: str) -> str:
    return (
        f"User message: {message}\n\n"
        f"Agent reply to evaluate:\n{reply}"
    )


# ── Evaluation model ──────────────────────────────────────────────────────────

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str


# ── Hard character cleaner ───────────────────────────────────────────────────
# Runs on every reply BEFORE evaluation — strips forbidden chars regardless of
# what the LLM produced. Evaluator then catches anything logic-based (tone, facts, etc.)

def _clean_reply(text: str) -> str:
    # Remove markdown syntax characters
    text = re.sub(r'[*#_`~]', '', text)
    # Remove emojis and all non-ASCII characters
    text = re.sub(r'[^\x20-\x7E]', '', text)
    # Collapse multiple spaces left by removals
    text = re.sub(r' {2,}', ' ', text).strip()
    return text


# ── Pushover helper ───────────────────────────────────────────────────────────

def _push(text: str) -> None:
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


# ── Tool functions ────────────────────────────────────────────────────────────

def get_project_details(names: list[str]) -> dict:
    """Direct project lookup by name. Fast — no embedding needed."""
    return {"result": get_by_names(names)}


def search_projects(description: str, top_n: int = DEFAULT_TOP_N) -> dict:
    """
    Semantic FAISS search over project search_text fields.
    Fallback tool — used when LLM has a description but not a project name.
    """
    return {"result": _agent.faiss_search(description, top_n)}


def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided") -> dict:
    _push(f"Portfolio visitor: {name} | email: {email} | notes: {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question: str) -> dict:
    _push(f"Unanswered question on portfolio: {question}")
    return {"recorded": "ok"}


# ── Tool schemas ──────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_project_details",
            "description": (
                "Get full details for one or more projects by name. "
                "Use this when the user mentions a project by name or you know exactly "
                "which project(s) they are asking about. "
                "Known projects: Sofia, MotionX, Groceria, QuickRef, AnoNote, CropDisease."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of project names, e.g. [\"Sofia\", \"Groceria\"]",
                    }
                },
                "required": ["names"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_projects",
            "description": (
                "Search for projects using a semantic description when you don't know "
                "the specific project name. Use this as a fallback when the user describes "
                "something like 'games you built', 'computer vision projects', "
                "'multi-agent systems', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A description of what the user is looking for",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "How many projects to return (default 3, max 6)",
                        "default": 3,
                    },
                },
                "required": ["description"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Record that a visitor wants to stay in touch and has provided their email address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The visitor's email address"},
                    "name":  {"type": "string", "description": "The visitor's name, if provided"},
                    "notes": {"type": "string", "description": "Any useful context about the conversation"},
                },
                "required": ["email"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Record any question you could not answer because you didn't know the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question that couldn't be answered"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
]


# ── Agent ─────────────────────────────────────────────────────────────────────

class SulithaAgent:

    def __init__(self):
        self.name   = "Sulitha Nulaksha Bandara"
        self.openai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

        # Build FAISS index once at startup
        logger.info("Building project FAISS index...")
        search_entries     = get_all_search_texts()
        self._project_keys = [k for k, _ in search_entries]
        texts              = [t for _, t in search_entries]

        sync_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        response    = sync_client.embeddings.create(model=EMBED_MODEL, input=texts)
        vecs        = np.array([d.embedding for d in response.data], dtype=np.float32)

        self._index = faiss.IndexFlatL2(vecs.shape[1])
        self._index.add(vecs)
        logger.info(f"Project FAISS index ready: {self._index.ntotal} projects, dim={vecs.shape[1]}")

    # ── FAISS search ──────────────────────────────────────────────────────────

    def faiss_search(self, description: str, top_n: int = DEFAULT_TOP_N) -> str:
        sync_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        response    = sync_client.embeddings.create(model=EMBED_MODEL, input=[description])
        q_vec       = np.array([response.data[0].embedding], dtype=np.float32)

        top_n    = min(top_n, len(self._project_keys))
        _, idxs  = self._index.search(q_vec, top_n)

        matched = [self._project_keys[i] for i in idxs[0] if i < len(self._project_keys)]
        return get_by_names(matched)

    # ── System prompt ─────────────────────────────────────────────────────────

    def _system_prompt(self) -> str:
        return (
            f"You are {self.name}, chatting casually with visitors on your portfolio site.\n\n"
            f"Tone rules:\n"
            f"- Talk like a normal, friendly person. Short and natural, not formal.\n"
            f"- Visitors are already on your portfolio — they know who you are. Never introduce yourself unprompted.\n"
            f"- If someone just says hi or hello, reply with a short warm greeting and offer to help. Nothing else.\n"
            f"- Never mention you are a third-year student, from Sri Lanka, or any other bio detail unless directly asked.\n"
            f"- Never say things like 'Greetings', 'Thank you for visiting', 'How may I assist you'.\n"
            f"- By default reply in 1-2 sentences. Only expand if the visitor clearly wants detail.\n"
            f"- If a topic has a lot of detail, give a short summary and end with something like "
            f"'want me to go deeper on any part?' — do not dump everything at once.\n\n"
            f"Tool rules:\n"
            f"- If a visitor asks about a specific project, call get_project_details.\n"
            f"- If they describe something without naming a project, call search_projects.\n"
            f"- If you cannot answer something, call record_unknown_question.\n"
            f"- If a visitor seems interested in getting in touch, ask for their email and "
            f"call record_user_details.\n\n"
            f"## Your Profile\n{PROFILE_SUMMARY}"
        )

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    def handle_tool_call(self, tool_calls) -> list[dict]:
        results = []
        for tc in tool_calls:
            name      = tc.function.name
            arguments = json.loads(tc.function.arguments)
            logger.info(f"Tool called: {name}({arguments})")
            fn     = globals().get(name)
            result = fn(**arguments) if fn else {"error": f"unknown tool: {name}"}
            results.append({
                "role":         "tool",
                "content":      json.dumps(result),
                "tool_call_id": tc.id,
            })
        return results

    # ── Response evaluation ───────────────────────────────────────────────────

    async def _evaluate(self, reply: str, message: str) -> Evaluation:
        """Use gpt-4.1-nano to evaluate the reply against quality rules."""
        response = await self.openai.beta.chat.completions.parse(
            model           = EVAL_MODEL,
            messages        = [
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user",   "content": _evaluator_user_prompt(reply, message)},
            ],
            response_format = Evaluation,
        )
        return response.choices[0].message.parsed

    async def _rerun(self, reply: str, message: str, history: list, feedback: str) -> str:
        """Ask GPT-4o-mini to fix the reply based on evaluator feedback."""
        fix_instruction = (
            f"Your previous reply did not meet the quality rules.\n"
            f"Feedback: {feedback}\n\n"
            f"Previous reply: {reply}\n\n"
            f"Please rewrite the reply fixing all issues mentioned in the feedback. "
            f"Remember: plain text only, no markdown, no emojis, maximum 60 words, "
            f"casual friendly tone, never formal greetings."
        )
        messages = (
            [{"role": "system", "content": self._system_prompt()}]
            + history
            + [{"role": "user",      "content": message}]
            + [{"role": "assistant", "content": reply}]
            + [{"role": "user",      "content": fix_instruction}]
        )
        response = await self.openai.chat.completions.create(
            model       = OPENAI_MODEL,
            messages    = messages,
            max_tokens  = MAX_TOKENS,
            temperature = 0.3,   # lower temp for correction pass
        )
        return _clean_reply(response.choices[0].message.content)

    # ── Main chat ─────────────────────────────────────────────────────────────

    async def chat(self, message: str, session_id: str) -> str:
        if session_store.is_over_limit(session_id):
            return SESSION_LIMIT_REPLY

        history  = session_store.get_history(session_id)
        messages = (
            [{"role": "system", "content": self._system_prompt()}]
            + history
            + [{"role": "user", "content": message}]
        )

        # 1. Main agent loop (handles tool calls)
        response = None
        try:
            while True:
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
                    break
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return (
                "Sorry, I am having a technical issue right now. "
                "Feel free to email me directly at nulakshastudy19@gmail.com."
            )

        reply = _clean_reply(response.choices[0].message.content)

        # 2. Evaluate and rerun if needed
        try:
            evaluation = await self._evaluate(reply, message)
            if evaluation.is_acceptable:
                logger.info("Evaluation passed.")
            else:
                logger.info(f"Evaluation failed — retrying. Feedback: {evaluation.feedback}")
                reply = await self._rerun(reply, message, history, evaluation.feedback)
        except Exception as e:
            logger.warning(f"Evaluation step failed — using original reply. Error: {e}")

        # 3. Save and return
        session_store.append(session_id, "user",      message)
        session_store.append(session_id, "assistant", reply)
        return reply


# ── Module-level interface (used by app.py) ───────────────────────────────────

_agent: SulithaAgent | None = None


def build_index() -> None:
    global _agent
    _agent = SulithaAgent()


async def chat(message: str, session_id: str) -> str:
    return await _agent.chat(message, session_id)
