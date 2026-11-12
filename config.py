import logging

# One place to change the topic of the whole project.
# Edit these values to switch the assistant to another subject.

APP_TITLE = "Civil Engineering Document Assistant"

APP_DESCRIPTION = (
    "Demo assistant that answers questions about concrete, hydraulics and "
    "water resources engineering using the PDF documents in its knowledge base."
)

ASSISTANT_ROLE = (
    "You are a technical assistant for civil and water resources engineers."
)

NOT_ENOUGH_INFO = (
    "I don't have enough information in my documents to answer that."
)

DISCLAIMER = (
    "Demo only. Answers come from the uploaded documents, which may be outdated "
    "or may not match your local codes and standards. Verify against the governing "
    "standards, and have a licensed engineer check any design decision. "
    "Not for design sign-off."
)

EXAMPLE_QUESTIONS = [
    "What factors affect the compressive strength of concrete?",
    "How should freshly placed concrete be cured?",
    "What is the purpose of a slump test?",
]

# Name of the ChromaDB collection (changing it avoids mixing old and new data).
COLLECTION_NAME = "civil-engineering-docs"

# Show the retrieved passages under the answer (useful while testing;
# set to False before sharing the demo widely).
SHOW_RETRIEVED_PASSAGES = True


def setup_logging(level: int = logging.INFO) -> None:
    """Call once, at the very start of the program (before importing rag_engine)."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    # Quiet chatty libraries (they log every web request at INFO level).
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
        