"""FastAPI entrypoint for Phase 1 scaffold.
- /chat  : Accepts JSON { "user_input": "..." } and returns structured requirement
- This file wires the conversational agent, scraper, RAG stub, and test plan generator.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modules.conversational_agent import ConversationalAgent
from modules.web_scraper import WebScraper
from modules.test_generator import TestGenerator
import uvicorn
import os

app = FastAPI(title="AI QA Agent - Phase1")

# Initialize core components (singletons for MVP)
agent = ConversationalAgent()
scraper = WebScraper()
generator = TestGenerator()

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        # Step 1: Parse requirement using LLM wrapper
        structured = agent.parse_requirement(req.user_input)

        # If a URL present, do a quick site scan
        url = structured.get("details", {}).get("url")
        if url:
            scraped = scraper.quick_scan(url)
            structured['scraped_summary'] = scraped

        # Generate a test plan
        plan = generator.generate_plan(structured)

        return {"status": "ok", "structured": structured, "plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('PORT', 8000)))
