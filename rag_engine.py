import logging
import math
import os
import re
import time
from collections import Counter

import chromadb
from google import genai
from google.genai import errors
from sentence_transformers import SentenceTransformer

import config

if __name__ == "__main__":
    config.setup_logging()  # only when run directly: python rag_engine.py

from pdf_loader import load_pdf_documents  # noqa: E402

logger = logging.getLogger(__name__)

# Every PDF found in the ./pdfs folder, split into chunks
ALL_DOCUMENTS = load_pdf_documents()
ID_TO_INDEX = {doc["id"]: i for i, doc in enumerate(ALL_DOCUMENTS)}

# 1. Load API key securely from environment (a Secret on Hugging Face)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY environment variable before running.")

# Main model, plus backups that are tried if it is overloaded or fails.
# Change them without editing code: add GEMINI_MODEL / GEMINI_FALLBACK_MODELS
# (comma-separated) as variables in the Space settings.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash,gemini-2.5-flash").split(",")
    if m.strip() and m.strip() != MODEL_NAME
]

# 2. Embedding model + vector DB
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)

# 3. Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


# 4. Keyword index (BM25): finds chunks that share the exact words of a question
def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class KeywordIndex:
    def __init__(self, docs, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        tokens = [_tokenize(d["content"]) for d in docs]
        self.doc_len = [len(t) for t in tokens]
        self.avg_len = (sum(self.doc_len) / len(docs)) if docs else 1.0
        self.tf = [Counter(t) for t in tokens]
        df = Counter()
        for t in tokens:
            df.update(set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def search(self, query: str, top_k: int):
        words = [w for w in _tokenize(query) if w in self.idf]
        scored = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            for w in words:
                f = tf.get(w, 0)
                if f:
                    norm = 1 - self.b + self.b * self.doc_len[i] / self.avg_len
                    score += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:top_k]]


KEYWORD_INDEX = KeywordIndex(ALL_DOCUMENTS)


# 5. Index documents (call this once at startup; safe to call repeatedly)
def index_documents(batch_size: int = 500):
    total = 0
    for start in range(0, len(ALL_DOCUMENTS), batch_size):
        batch = ALL_DOCUMENTS[start:start + batch_size]
        texts = [doc["content"] for doc in batch]
        collection.upsert(  # upsert = insert new, update existing (no duplicate errors)
            embeddings=embedder.encode(texts).tolist(),
            documents=texts,
            metadatas=[{"title": doc["title"]} for doc in batch],
            ids=[doc["id"] for doc in batch],
        )
        total += len(batch)
        logger.info("Embedded %d of %d chunks", total, len(ALL_DOCUMENTS))
    logger.info("Indexed %d text chunks", total)


def knowledge_base_summary() -> str:
    """Markdown list of the loaded PDFs and how many chunks each produced."""
    counts = Counter(doc["source"] for doc in ALL_DOCUMENTS)
    if not counts:
        return "No documents loaded. Add PDF files to the 'pdfs' folder."
    lines = [f"- **{name}**: {n} chunks" for name, n in counts.items()]
    return "\n".join(lines) + f"\n\nTotal: {len(ALL_DOCUMENTS)} chunks"


# 6. Retrieve: hybrid search = meaning (embeddings) + exact words (BM25)
CANDIDATES = 20  # how many results each method contributes before merging
RRF_K = 60       # standard constant for reciprocal rank fusion


def retrieve(query: str, top_k: int = 6):
    """Return the best chunks as dicts with 'title' and 'content'."""
    n = collection.count()
    if n == 0:
        return []

    query_embedding = embedder.encode([query]).tolist()
    vector_hits = collection.query(
        query_embeddings=query_embedding,
        n_results=min(CANDIDATES, n),
    )["ids"][0]
    keyword_hits = KEYWORD_INDEX.search(query, CANDIDATES)

    # Reciprocal rank fusion: chunks ranked high by either method float to the top.
    scores = {}
    for rank, doc_id in enumerate(vector_hits):
        i = ID_TO_INDEX.get(doc_id)
        if i is not None:
            scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, i in enumerate(keyword_hits):
        scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)

    best = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [ALL_DOCUMENTS[i] for i in best]


# 7. Generate answer
BUSY_MESSAGE = "The AI service is busy right now, please try again in a moment."
QUOTA_MESSAGE = (
    "The Gemini usage limit has been reached for now. Please wait a minute and try "
    "again. If it keeps happening, the daily free limit may be used up."
)
KEY_MESSAGE = (
    "The Gemini API key was rejected. Check that the GEMINI_API_KEY secret holds "
    "a valid, active key."
)
ERROR_MESSAGE = "Sorry, something went wrong while generating the answer."


def _classify(error) -> str:
    """Turn a Gemini error into one of: quota, key, busy, other."""
    code = getattr(error, "code", None)
    text = str(error).lower()
    if code == 429 or "resource_exhausted" in text or "quota" in text:
        return "quota"
    if code in (401, 403) or "api key" in text or "api_key" in text or "permission" in text:
        return "key"
    if isinstance(error, errors.ServerError):
        return "busy"
    return "other"


def generate_with_retry(prompt, retries=3):
    """Try the main model, then each backup model. Every failure is logged."""
    kinds = []
    started = time.time()
    for model in [MODEL_NAME] + FALLBACK_MODELS:
        for attempt in range(retries):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                if response.text:
                    logger.info("Answered by %s in %.1fs", model, time.time() - started)
                    return response.text
                logger.warning("[%s] returned an empty answer", model)
                break  # try the next model
            except errors.ServerError as e:  # overloaded / temporary problem
                kinds.append("busy")
                logger.warning("[%s] server busy (attempt %d/%d): %s", model, attempt + 1, retries, e)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s
            except Exception as e:  # wrong model name, bad key, quota, etc.
                kinds.append(_classify(e))
                logger.warning("[%s] error (%s): %s", model, kinds[-1], e)
                break  # retrying won't help; try the next model
    logger.error("All models failed: %s", kinds)
    # Report the most useful cause (a later backup model may fail for a
    # different reason, such as an unknown model name).
    for kind, message in (("quota", QUOTA_MESSAGE), ("key", KEY_MESSAGE), ("busy", BUSY_MESSAGE)):
        if kind in kinds:
            return message
    return ERROR_MESSAGE


def generate_answer(query: str, retrieved_docs: list, titles: list) -> str:
    context = "\n\n".join(
        f"[Source: {title}]\n{content}"
        for title, content in zip(titles, retrieved_docs)
    )

    prompt = f"""{config.ASSISTANT_ROLE}
Answer the question using ONLY the provided context. If the context doesn't contain the answer, say
"{config.NOT_ENOUGH_INFO}"

Quote numbers, units, and table values exactly as written in the context.
Do not estimate or invent values, and do not do calculations the context does not show.

Context:
{context}

Question: {query}

Answer concisely and cite sources (document name and page) where possible."""
    return generate_with_retry(prompt)


# 8. Full pipeline
def ask(query: str):
    if collection.count() == 0:
        return {
            "answer": "The knowledge base is empty. Add PDF files to the 'pdfs' folder.",
            "sources": [],
            "passages": [],
        }
    logger.info("Question: %s", query[:100])
    started = time.time()
    chunks = retrieve(query)
    logger.info(
        "Retrieved %d chunks in %.2fs from: %s",
        len(chunks), time.time() - started, sorted({c["title"] for c in chunks}),
    )
    docs = [c["content"] for c in chunks]
    titles = [c["title"] for c in chunks]
    answer = generate_answer(query, docs, titles)
    return {"answer": answer, "sources": titles, "passages": docs}


# 9. Test (only when run directly: python rag_engine.py)
if __name__ == "__main__":
    index_documents()
    test_query = config.EXAMPLE_QUESTIONS[0]
    result = ask(test_query)
    print(f"Q: {test_query}")
    print(f"A: {result['answer']}")
    print(f"Sources: {result['sources']}")
