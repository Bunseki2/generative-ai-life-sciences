import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # Week 09 — Biomedical literature RAG assistant

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
    import pandas as pd

    # Define an auditable corpus schema. Use PubMed/Europe PMC or other permitted sources.
    # Preserve source identifiers, URLs, and licenses with every document passage.
    corpus_columns = ["doc_id", "title", "year", "source_url", "license", "passage"]
    corpus = pd.DataFrame(columns=corpus_columns)
    print("Required corpus schema:")
    print(corpus)

    # TODO 1 — Ingest:
    # Download a small set of permitted abstracts/open-access full-text documents.
    # Respect API policies and reuse licenses. Record PMID/DOI, title, year, URL, license.
    #
    # TODO 2 — Chunk:
    # Split long documents into overlapping passages (e.g., 400-700 tokens).
    # Preserve doc_id and source metadata for each chunk.
    #
    # TODO 3 — Retrieve:
    # Reuse the Week 6 embedding and FAISS approach.
    # Return passage text AND source metadata; compare with keyword retrieval.
    # Evaluate Recall@k using questions with known relevant PMIDs.
    #
    # TODO 4 — Generate:
    # Connect a local model or an approved inference endpoint; do not hard-code API keys.
    # Prompt it to answer only from retrieved passages, cite every factual claim with
    # [doc_id], and abstain when evidence is missing or conflicting.
    # Distinguish experimental findings, author hypotheses, and assistant synthesis.
    #
    # TODO 5 — Evaluate:
    # Use the Week 8 rubric for relevance, factual support, citation correctness,
    # abstention behavior, and hallucinations.
    return


@app.cell
def _():
    import marimo as mo
    mo.md("""
    ## GitHub deliverable

    Complete the TODOs, add your own explanations and plots, and save a short
    `reports/week09_summary.md`.

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
