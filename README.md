---
title: Sulitha Portfolio Agent
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# Sulitha's Portfolio Agent

A FastAPI agent hosted on a HuggingFace Docker Space.
Visitors chat on the portfolio site — the agent responds as Sulitha,
knowing his projects, skills, research, and availability.

---

## Security Model

Requests flow through a Vercel API route proxy — the shared secret never
reaches the browser:

```
Browser --> Vercel /api/chat (secret in server env) --> HF Space /chat
```

Layers:
- CORS — portfolio domain only
- X-Portfolio-Key header — set server-side by Vercel proxy, never in browser JS
- Rate limiting — 10 requests/min per IP (slowapi)
- Session message cap — max 20 messages then redirects to email

## Agent Design

Tools (real OpenAI function calling):
- `record_user_details` — fires when a visitor shares their email (Pushover notification)
- `record_unknown_question` — fires when a question can't be answered (Pushover notification)

RAG: knowledge.py chunks embedded at startup into a FAISS index.
Top-3 relevant chunks are injected into the system prompt for each message.

## Environment Variables (set in Space Secrets)

| Variable             | Required | Purpose                        |
|----------------------|----------|--------------------------------|
| OPENAI_API_KEY       | Yes      | GPT-4o-mini calls              |
| PORTFOLIO_SECRET_KEY | Yes      | Must match Vercel env var      |
| PUSHOVER_TOKEN       | Optional | Phone notifications            |
| PUSHOVER_USER        | Optional | Phone notifications            |

## Stack

FastAPI, OpenAI GPT-4o-mini, FAISS, sentence-transformers/all-MiniLM-L6-v2,
slowapi, Docker, HuggingFace Spaces.
