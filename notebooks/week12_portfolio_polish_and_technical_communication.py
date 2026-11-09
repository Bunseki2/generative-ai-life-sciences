import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # Week 12 — Portfolio polish and technical communication

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
    project_inventory = [
        {"project": "PyTorch molecular baseline", "status": "TODO", "evidence": "Weeks 1-2"},
        {"project": "Biomedical semantic search / RAG", "status": "TODO", "evidence": "Weeks 6 and 9"},
        {"project": "Protein language model embeddings", "status": "TODO", "evidence": "Week 10"},
        {"project": "CYP evidence assistant", "status": "TODO", "evidence": "Week 11"},
    ]
    print(project_inventory)

    # TODO — Reproducibility:
    # - Add environment/package versions and setup instructions.
    # - Document dataset source, license, preprocessing, and split strategy.
    # - Report baselines, metrics, random seeds, and model versions.
    # - Ensure notebooks run top-to-bottom from a clean Molab session.
    # - Remove secrets, private data, and large model checkpoints.
    #
    # TODO — Scientific communication:
    # For each selected project, write problem, data, methods, baseline, results,
    # limitations, and what you personally implemented versus adapted.
    #
    # TODO — Job readiness:
    # - Prepare a 60-second explanation for a scientific audience.
    # - Prepare a 30-second explanation for a non-specialist.
    # - Prepare one debugging story, one evaluation decision, and one limitation found.
    # - Update resume/LinkedIn only with tools and methods you can demonstrate.
    return


@app.cell
def _():
    import marimo as mo
    mo.md("""
    ## GitHub deliverable

    Complete the TODOs, add your own explanations and plots, and save a short
    `reports/week12_summary.md`.

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
