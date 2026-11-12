# IMPORTANT: `spaces` must be imported BEFORE anything that loads PyTorch
# (sentence-transformers, rag_engine, ...), so it stays at the very top.
# ZeroGPU Spaces expect at least one function decorated with @spaces.GPU.
# This app runs on CPU, so the function below is never called; it only
# satisfies the startup check. Locally (no `spaces` package) it is skipped.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return "ok"

except ImportError:
    pass

import gradio as gr

import config

config.setup_logging()  # must run before rag_engine is imported

from rag_engine import ask, index_documents, knowledge_base_summary  # noqa: E402

# Build the vector index once, when the app starts.
index_documents()


def answer_question(question: str):
    question = (question or "").strip()
    if not question:
        return "Please type a question.", ""

    result = ask(question)

    # Remove duplicate titles while keeping their order.
    unique_sources = list(dict.fromkeys(result["sources"]))
    answer_md = f"### Answer\n{result['answer']}"
    if unique_sources:
        sources_md = "\n".join(f"- {title}" for title in unique_sources)
        answer_md += f"\n\n### Sources\n{sources_md}"

    passages_text = "\n\n".join(
        f"[{title}]\n{text}"
        for title, text in zip(result["sources"], result.get("passages", []))
    )
    return answer_md, passages_text


with gr.Blocks(title=config.APP_TITLE) as demo:
    gr.Markdown(f"# {config.APP_TITLE}")
    gr.Markdown(f"{config.APP_DESCRIPTION}\n\n**Disclaimer:** {config.DISCLAIMER}")

    question = gr.Textbox(
        lines=3,
        label="Question",
        placeholder=f"e.g. {config.EXAMPLE_QUESTIONS[0]}",
    )
    submit = gr.Button("Submit", variant="primary")
    answer = gr.Markdown()

    with gr.Accordion(
        "Retrieved passages (what the AI was given)",
        open=False,
        visible=config.SHOW_RETRIEVED_PASSAGES,
    ):
        passages = gr.Textbox(lines=12, show_label=False, interactive=False)

    with gr.Accordion("Loaded documents", open=False):
        gr.Markdown(knowledge_base_summary())

    gr.Examples(examples=[[q] for q in config.EXAMPLE_QUESTIONS], inputs=question)

    submit.click(answer_question, inputs=question, outputs=[answer, passages])

# Limit the waiting line so one visitor can't burn through the Gemini quota.
demo.queue(max_size=20)

if __name__ == "__main__":
    demo.launch()
