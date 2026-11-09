import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # Week 04 — Transformer inference and embeddings

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
    import pandas as pd

    model_id = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder = AutoModel.from_pretrained(model_id).eval()

    texts = [
        "Amyloid beta aggregation is associated with fibril formation.",
        "Peptide self-assembly can be studied using molecular dynamics.",
        "CYP3A4 contributes to metabolism of many therapeutic compounds.",
        "Cytochrome P450 enzymes metabolize xenobiotics.",
    ]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = encoder(**inputs)
        token_embeddings = output.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        mean_embeddings = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1)
        cls_embeddings = token_embeddings[:, 0, :]

    mean_sim = cosine_similarity(mean_embeddings.cpu().numpy())
    cls_sim = cosine_similarity(cls_embeddings.cpu().numpy())
    print("Mean-pooled similarity:")
    print(pd.DataFrame(mean_sim, index=range(len(texts)), columns=range(len(texts))))
    print("CLS similarity:")
    print(pd.DataFrame(cls_sim, index=range(len(texts)), columns=range(len(texts))))

    # TODO:
    # - Compare mean pooling and CLS pooling.
    # - Explain why padding tokens must not contribute to mean pooling.
    # - Inspect whether similar sentences are scientifically related.
    # - Explain why cosine similarity is not proof of factual equivalence.
    return


@app.cell
def _():
    import marimo as mo
    mo.md("""
    ## GitHub deliverable

    Complete the TODOs, add your own explanations and plots, and save a short
    `reports/week04_summary.md`.

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
