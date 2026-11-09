import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # Week 11 — Scientific AI assistant: CYP evidence prototype

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
    from pydantic import BaseModel
    from typing import Literal, Optional

    class CYPRecord(BaseModel):
        compound_id: str
        smiles: Optional[str] = None
        isoform: Literal["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
        evidence_type: Literal["assay_measurement", "model_prediction", "literature_statement"]
        value: Optional[float] = None
        unit: Optional[str] = None
        source: str
        notes: Optional[str] = None

    example = CYPRecord(
        compound_id="example-001",
        isoform="CYP3A4",
        evidence_type="model_prediction",
        value=0.72,
        unit="illustrative score",
        source="synthetic demonstration",
        notes="Not an experimental result",
    )
    print(example.model_dump())

    # TODO:
    # 1. Load your processed OpenADMET CYP Challenge results or model predictions.
    # 2. Preserve exact target names, units, confidence bounds, and provenance.
    # 3. Implement filtering by isoform AND evidence_type.
    # 4. Return source records before generating any natural-language summary.
    # 5. Add tests ensuring a model prediction is never labeled as measured IC50/pIC50.
    # 6. Test an unanswerable question; the correct response should say evidence is missing.
    return


@app.cell
def _():
    import marimo as mo
    mo.md("""
    ## GitHub deliverable

    Complete the TODOs, add your own explanations and plots, and save a short
    `reports/week11_summary.md`.

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
