# AI QA Agent — Phase 1 Starter Scaffold

This scaffold provides a minimal, well-commented starter repository for the **Phase 1 (MVP)**:
- Conversational (chat) layer (FastAPI stub)
- RAG ingestion skeleton (LangChain/LlamaIndex style placeholders)
- Web scraper (Playwright) that saves markdown
- Test plan generator (LLM prompt stub)
- Config, logging, and run instructions

**Goal:** Run locally on macOS (Intel). This scaffold is intentionally dependency-light:
- It uses environment variables for LLM keys (OpenAI/Anthropic) if available.
- Where external services are needed, stubs or safe fallbacks are provided.

## Quick start (macOS)
1. Clone or unzip this scaffold.
2. Create and activate a Python 3.10+ virtualenv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Install Playwright browsers:
   ```bash
   playwright install
   ```
4. Export your OpenAI API key (optional):
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
5. Run the FastAPI app (conversational endpoint):
   ```bash
   uvicorn main:app --reload
   ```
6. Use the `/chat` endpoint (POST) to send a requirement JSON, or run `scripts/demo_run.py` for a local demo.

## Structure
- `main.py` - FastAPI server with `/chat` endpoint (conversation stub)
- `modules/conversational_agent.py` - Conversational agent wrapper (LLM calls)
- `modules/rag_engine.py` - RAG ingestion & retrieval skeleton
- `modules/web_scraper.py` - Playwright-based site scanner that saves markdown
- `modules/test_generator.py` - Converts plan -> test plan JSON (LLM prompt stub)
- `modules/runner.py` - Executes generated smoke tests (placeholder)
- `config/config.yaml` - Basic project config
- `scripts/demo_run.py` - Demo flow: chat -> scrape -> generate plan -> run small smoke
- `requirements.txt` - Python deps
- `Dockerfile`, `docker-compose.yml` - Optional containerization

## Notes
- This is a **Phase 1 MVP** scaffold — it focuses on correctness, comments, and a runnable local PoC.
- The LLM calls are abstracted; you can wire OpenAI, Anthropic, or local Llama endpoints by setting environment variables.
- After validating the flow, we will extend to framework generation (Playwright tests), RAG tuning, and self-healing.

If you want, I can now walk you through running `scripts/demo_run.py` step-by-step on your machine.
