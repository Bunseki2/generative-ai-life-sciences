import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # Week 08 — LLM evaluation and error analysis

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

    cases = pd.DataFrame([
        {"id": "Q1", "question": "What is a model prediction?",
         "required_fact": "A computational estimate, not an experimental measurement."},
        {"id": "Q2", "question": "Does high sequence similarity prove identical protein function?",
         "required_fact": "No; similarity alone does not establish identical function."},
        {"id": "Q3", "question": "What does a citation support?",
         "required_fact": "The cited source should directly support the associated claim."},
        {"id": "Q4", "question": "Can a predicted CYP score be called an experimental IC50?",
         "required_fact": "No; predictions and experimental measurements must be distinguished."},
        {"id": "Q5", "question": "What should an assistant do when evidence is missing?",
         "required_fact": "State that evidence is insufficient and abstain from unsupported claims."},
    ])
    results = cases.copy()
    results["model_answer"] = ""
    results["relevance_1_to_5"] = pd.NA
    results["factuality_1_to_5"] = pd.NA
    results["source_supported_yes_no"] = pd.NA
    results["fabricated_citation_yes_no"] = pd.NA
    results["error_notes"] = ""
    print(results.to_string(index=False))

    # TODO:
    # - Add answers from your base model, fine-tuned model, and/or RAG assistant.
    # - Score each answer using a written rubric (1=poor, 3=partial, 5=strong).
    # - Count unsupported claims and fabricated citations separately.
    # - Expand this set to 30-50 questions; start with 10 and iterate.
    # - Report model version, prompt, decoding settings, date, and reviewer.
    return


@app.cell
def _():
    import marimo as mo
    mo.md("""
    ## GitHub deliverable

    Complete the TODOs, add your own explanations and plots, and save a short
    `reports/week08_summary.md`.

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
