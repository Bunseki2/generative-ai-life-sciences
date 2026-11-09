import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # Week 06 — Datasets, embeddings, and semantic search

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
    import numpy as np
    import torch
    import faiss
    from transformers import AutoTokenizer, AutoModel

    documents = [
        {"id": "D1", "text": "Amyloid beta peptides aggregate and form cross-beta fibrils."},
        {"id": "D2", "text": "Molecular dynamics can characterize conformational changes in proteins."},
        {"id": "D3", "text": "CYP2D6 is a cytochrome P450 enzyme involved in drug metabolism."},
        {"id": "D4", "text": "Protein language models encode information from amino acid sequences."},
        {"id": "D5", "text": "Ligand binding can alter the conformational ensemble of a protein."},
        {"id": "D6", "text": "CYP3A4 metabolizes many drugs and other xenobiotic compounds."},
    ]
    encoder_id = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(encoder_id)
    encoder = AutoModel.from_pretrained(encoder_id).eval()

    def embed_texts(text_list):
        batch = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            hidden = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype("float32")

    corpus_vectors = embed_texts([d["text"] for d in documents])
    index = faiss.IndexFlatIP(corpus_vectors.shape[1])
    index.add(corpus_vectors)

    query = "Which enzyme helps metabolize medicines?"
    scores, ids = index.search(embed_texts([query]), k=3)
    retrieved = [
        {"id": documents[i]["id"], "text": documents[i]["text"], "score": float(score)}
        for i, score in zip(ids[0], scores[0])
    ]
    print(retrieved)

    # TODO: Create 5 queries and manually label relevant document IDs.
    # Calculate Recall@k for k=1, 3, and compare with a keyword-search baseline.
    # Note: semantic similarity is a retrieval signal, not evidence of factual correctness.
    return


@app.cell
def _():
    import marimo as mo
    mo.md("""
    ## GitHub deliverable

    Complete the TODOs, add your own explanations and plots, and save a short
    `reports/week06_summary.md`.

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
