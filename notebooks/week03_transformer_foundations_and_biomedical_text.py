import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # Week 03 — Transformer foundations and biomedical text

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
    from transformers import AutoTokenizer

    texts = [
        "Amyloid beta peptides can aggregate into fibrillar structures.",
        "Molecular dynamics simulations provide atomistic views of protein motion.",
        "Cytochrome P450 enzymes are involved in xenobiotic metabolism.",
        "Protein language models learn representations from amino acid sequences.",
    ]
    model_id = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    print("Tensor shapes:", {k: tuple(v.shape) for k, v in batch.items()})
    for text, ids, mask in zip(texts, batch["input_ids"], batch["attention_mask"]):
        print("\nTEXT:", text)
        print("TOKENS:", tokenizer.convert_ids_to_tokens(ids))
        print("ATTENTION MASK:", mask.tolist())

    # TODO: Write Markdown answers:
    # 1. What do input_ids and attention_mask represent?
    # 2. Why are special tokens inserted?
    # 3. What happens when a text is truncated?
    # 4. Why might biomedical-domain pretraining/tokenization help?
    # 5. Compare token counts for PubMedBERT and a general-purpose tokenizer.
    return


@app.cell
def _():
    import marimo as mo
    mo.md("""
    ## GitHub deliverable

    Complete the TODOs, add your own explanations and plots, and save a short
    `reports/week03_summary.md`.

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
