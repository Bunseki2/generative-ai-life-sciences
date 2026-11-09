import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # Week 10 — Protein language models with ESM-2

    **Estimated effort:** 10–15 hours, including lessons, coding, notes, and GitHub.

    **Molab setup:** Install missing packages using the Molab package manager.
    Do not use IPython `%pip` or `%conda` magic commands.

    **Scientific practice:** Record data provenance, licenses, random seeds, model
    versions, and limitations. Never upload confidential data or credentials.
    """)
    return


@app.cell
def _():
    # Main exercise. All variables are intentionally kept in this cell's scope.
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.metrics.pairwise import cosine_similarity

    # Short illustrative sequences are only for checking the embedding pipeline.
    # Replace with a documented protein dataset for a real analysis.
    sequences = [
        "MKWVTFISLLFLFSSAYS",
        "MKWVTFISLLFLFSSAYS",
        "GAVLILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG",
    ]
    model_id = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()

    batch = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**batch)
        hidden = output.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        embeddings = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    print("Embedding shape:", tuple(embeddings.shape))
    print(cosine_similarity(embeddings.cpu().numpy()))

    # TODO:
    # - Compare cosine similarities and explain the duplicate-sequence result.
    # - Discuss whether special tokens should be excluded from pooling.
    # - Obtain a labeled protein dataset for a downstream classifier.
    # - Use sequence-level splits that reduce homology leakage.
    # - Do not infer function, structure, or binding from embedding similarity alone.
    return


@app.cell
def _():
    import marimo as mo
    mo.md("""
    ## GitHub deliverable

    Complete the TODOs, add your own explanations and plots, and save a short
    `reports/week10_summary.md`.

    ## Completion checklist

    - [ ] Notebook runs top-to-bottom in a fresh Molab session.
    - [ ] I can explain the methods and design choices in my own words.
    - [ ] I recorded data/model provenance and relevant versions.
    - [ ] I evaluated limitations and at least one failure mode.
    - [ ] I committed the notebook and a short summary to GitHub.
    """)
    return


if __name__ == "__main__":
    app.run()
