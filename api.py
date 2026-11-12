from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

import rag_settings as config

config.setup_logging()  # must run before rag_engine is imported

from rag_engine import ask, index_documents  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: build the vector index.
    index_documents()
    yield


app = FastAPI(
    title=f"{config.APP_TITLE} API",
    description=config.APP_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Answer(BaseModel):
    answer: str
    sources: list[str]
    disclaimer: str = config.DISCLAIMER


@app.get("/health")
def health():
    """Simple check that the server is up."""
    return {"status": "ok"}


# A normal (non-async) function: FastAPI runs it in a worker thread, so the
# slow embedding and Gemini calls don't block other requests.
@app.post("/ask", response_model=Answer)
def ask_endpoint(payload: Question):
    result = ask(payload.question.strip())
    unique_sources = list(dict.fromkeys(result["sources"]))
    return Answer(answer=result["answer"], sources=unique_sources)
