import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # Week 02 — Molecular data and classical ML baseline

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
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # TODO: Replace this tiny illustrative dataset with a licensed dataset.
    # Required columns: SMILES and target. These values are synthetic placeholders.
    data = pd.DataFrame({
        "SMILES": ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCOCC", "CC(C)O",
                   "CC(=O)N", "CCOC", "C1CCCCC1", "CC(C)C"],
        "target": [1.2, 0.8, 2.1, 1.0, 1.5, 1.1, 0.9, 1.4, 1.8, 1.3],
    })
    print("Rows:", len(data))
    print(data.head().to_string(index=False))

    # Parse SMILES and calculate interpretable descriptors.
    records, invalid_rows = [], []
    for idx, row in data.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            invalid_rows.append(idx)
            continue
        records.append({
            "row_id": idx,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "MolWt": Descriptors.MolWt(mol),
            "MolLogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "target": row["target"],
        })
    features = pd.DataFrame(records)
    print("Invalid SMILES row indices:", invalid_rows)
    print(features.head().to_string(index=False))

    # Descriptor baseline. For a real dataset, use a scaffold split when appropriate.
    feature_cols = ["MolWt", "MolLogP", "TPSA", "HBD", "HBA"]
    X = features[feature_cols]
    y = features["target"]
    train_idx, val_idx = train_test_split(
        np.arange(len(features)), test_size=0.3, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[val_idx])
    descriptor_metrics = {
        "MAE": mean_absolute_error(y.iloc[val_idx], pred),
        "RMSE": mean_squared_error(y.iloc[val_idx], pred) ** 0.5,
        "R2": r2_score(y.iloc[val_idx], pred),
    }
    print("Descriptor metrics (tiny synthetic example; not scientifically meaningful):")
    print(descriptor_metrics)

    # Morgan fingerprints.
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for smi in features["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smi)
        fps.append(fpgen.GetFingerprintAsNumPy(mol))
    X_fp = np.asarray(fps, dtype=np.float32)
    print("Fingerprint matrix:", X_fp.shape)

    # TODO: Train a second RandomForestRegressor on X_fp using train_idx/val_idx.
    # Compare descriptor and fingerprint results on the SAME split.
    # Stretch: compare against a simple MLP, then try a scaffold split.
    return


@app.cell
def _():
    import marimo as mo
    mo.md("""
    ## GitHub deliverable

    Complete the TODOs, add your own explanations and plots, and save a short
    `reports/week02_summary.md`.

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
