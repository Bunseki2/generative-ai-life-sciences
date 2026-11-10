import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md(r"""
    # Week 2 — ESOL molecular ML comparison

    Compare four combinations on the **ESOL (Delaney) aqueous-solubility
    dataset** (~1,128 molecules): Random Forest + RDKit descriptors, MLP +
    descriptors, Random Forest + Morgan fingerprints, and MLP + fingerprints.

    The original 10-molecule example is only for practicing featurization;
    it is too small for meaningful model comparisons.

    **Install in Molab if needed:**
    ```bash
    uv pip install pandas numpy scikit-learn torch rdkit marimo
    ```

    This notebook uses no IPython magic commands. It evaluates all four
    combinations under both a random split and a Bemis–Murcko scaffold split.
    Interpret those two test settings separately.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import random
    import urllib.request
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    path = Path("delaney-processed.csv")
    if not path.exists():
        urllib.request.urlretrieve(URL, path)

    raw = pd.read_csv(path)
    smiles_col = "smiles"
    target_col = "measured log solubility in mols per litre"
    data = raw[[smiles_col, target_col]].copy()
    data[smiles_col] = data[smiles_col].astype(str).str.strip()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna().reset_index(drop=True)
    data["mol"] = data[smiles_col].map(Chem.MolFromSmiles)
    invalid_count = int(data["mol"].isna().sum())
    data = data.dropna(subset=["mol"]).drop_duplicates(smiles_col).reset_index(drop=True)
    y = data[target_col].to_numpy(dtype=np.float32)

    mo.md(
        f"## 1. Load ESOL\n\n"
        f"Valid unique molecules: **{len(data):,}** · Invalid SMILES removed: "
        f"**{invalid_count}** · Device: **{DEVICE}**\n\n"
        f"Target: measured aqueous log-solubility (log mol/L)."
    )
    return (
        AllChem, Chem, DataLoader, DataStructs, DEVICE, Descriptors, Lipinski,
        MurckoScaffold, RandomForestRegressor, SEED, StandardScaler, TensorDataset,
        data, mean_absolute_error, mean_squared_error, nn, np, pd, r2_score,
        rdMolDescriptors, train_test_split, torch, y
    )


@app.cell
def _(AllChem, DataStructs, Descriptors, Lipinski, data, np, pd, rdMolDescriptors):
    import marimo as mo

    descriptor_fns = [
        ("MolWt", Descriptors.MolWt), ("MolLogP", Descriptors.MolLogP),
        ("TPSA", rdMolDescriptors.CalcTPSA), ("HBD", Lipinski.NumHDonors),
        ("HBA", Lipinski.NumHAcceptors), ("RotatableBonds", Lipinski.NumRotatableBonds),
        ("RingCount", Lipinski.RingCount), ("AromaticRings", Lipinski.NumAromaticRings),
        ("HeavyAtoms", Lipinski.HeavyAtomCount),
        ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
        ("FormalCharge", lambda mol: sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        ("LabuteASA", rdMolDescriptors.CalcLabuteASA),
    ]
    descriptor_names = [name for name, _ in descriptor_fns]
    X_desc = np.asarray(
        [[float(fn(m)) for _, fn in descriptor_fns] for m in data["mol"]],
        dtype=np.float32,
    )

    radius, n_bits = 2, 2048
    X_fp = np.zeros((len(data), n_bits), dtype=np.float32)
    for i, mol in enumerate(data["mol"]):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, X_fp[i])

    if not np.isfinite(X_desc).all() or not np.isfinite(X_fp).all():
        raise ValueError("Non-finite feature values found.")
    mo.md(
        f"## 2. RDKit features\n\n"
        f"Descriptors: **{X_desc.shape[1]}** · Morgan bits: **{n_bits}** "
        f"(radius {radius}; ECFP4-style)."
    )
    mo.ui.table(pd.DataFrame(X_desc, columns=descriptor_names).head(8), selection=None)
    return X_desc, X_fp


@app.cell
def _(MurckoScaffold, SEED, data, np, pd, train_test_split):
    import marimo as mo

    indices = np.arange(len(data))
    tr, temp = train_test_split(indices, test_size=0.20, random_state=SEED)
    va, te = train_test_split(temp, test_size=0.50, random_state=SEED)
    random_split = {"train": np.asarray(tr), "val": np.asarray(va), "test": np.asarray(te)}

    # Keep each Bemis–Murcko scaffold in exactly one partition. Empty scaffolds
    # (acyclic molecules) get unique group IDs instead of one shared empty group.
    groups = {}
    for i, mol in enumerate(data["mol"]):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        key = scaffold if scaffold else f"ACYCLIC_{i}"
        groups.setdefault(key, []).append(i)

    n = len(data)
    target_sizes = {"train": 0.80*n, "val": 0.10*n, "test": 0.10*n}
    assigned = {"train": [], "val": [], "test": []}
    for members in sorted(groups.values(), key=len, reverse=True):
        dest = max(target_sizes, key=lambda s: target_sizes[s] - len(assigned[s]))
        assigned[dest].extend(members)
    scaffold_split = {
        s: np.asarray(sorted(v), dtype=int) for s, v in assigned.items()
    }

    # Explicitly verify that no scaffold group crosses a partition.
    owner = {}
    for split, idxs in scaffold_split.items():
        for i in idxs:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=data.loc[int(i), "mol"], includeChirality=False
            )
            key = scaffold if scaffold else f"ACYCLIC_{i}"
            owner.setdefault(key, set()).add(split)
    assert all(len(splits) == 1 for splits in owner.values())

    summary = pd.DataFrame([
        {"strategy": name, "partition": part, "n": len(ix), "fraction": len(ix)/n}
        for name, split in [("Random", random_split), ("Scaffold", scaffold_split)]
        for part, ix in split.items()
    ])
    mo.md(
        "## 3. Train/validation/test partitions\n\n"
        "Random splitting can place close analogues in both train and test. "
        "Scaffold splitting keeps each Bemis–Murcko scaffold together, giving "
        "a more demanding chemotype-generalization test. Exact scaffold split "
        "fractions may differ from 80/10/10 because groups cannot be divided."
    )
    mo.ui.table(summary, selection=None)
    return random_split, scaffold_split


@app.cell
def _(
    DEVICE, DataLoader, RandomForestRegressor, SEED, StandardScaler,
    TensorDataset, X_desc, X_fp, mean_absolute_error, mean_squared_error,
    nn, np, pd, r2_score, random_split, scaffold_split, torch, y
):
    import marimo as mo

    class MLP(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.15),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.10),
                nn.Linear(128, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def metrics(y_true, y_pred):
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
        }

    def fit_mlp(X, split):
        it, iv, ix = split["train"], split["val"], split["test"]
        # Fit both scalers on training data only (no leakage).
        sx, sy = StandardScaler(), StandardScaler()
        xt = sx.fit_transform(X[it]).astype(np.float32)
        xv = sx.transform(X[iv]).astype(np.float32)
        xx = sx.transform(X[ix]).astype(np.float32)
        yt = sy.fit_transform(y[it, None]).ravel().astype(np.float32)
        yv = sy.transform(y[iv, None]).ravel().astype(np.float32)

        torch.manual_seed(SEED)
        model = MLP(X.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        ds = TensorDataset(torch.tensor(xt), torch.tensor(yt))
        loader = DataLoader(
            ds, batch_size=64, shuffle=True,
            generator=torch.Generator().manual_seed(SEED)
        )
        best, best_state, stale, patience = float("inf"), None, 0, 12
        epochs_run = 0
        for epoch in range(100):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred_val = model(torch.tensor(xv, device=DEVICE)).cpu().numpy()
            val_mae = mean_absolute_error(yv, pred_val)
            epochs_run = epoch + 1
            if val_mae < best - 1e-5:
                best = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(torch.tensor(xx, device=DEVICE)).cpu().numpy()
        return sy.inverse_transform(pred_scaled[:, None]).ravel(), epochs_run

    def run_experiments(split_name, split):
        rows = []
        for feature_name, X in [("descriptors", X_desc), ("Morgan fingerprints", X_fp)]:
            it, ix = split["train"], split["test"]
            rf = RandomForestRegressor(
                n_estimators=400, random_state=SEED, n_jobs=-1, max_features=1.0
            )
            rf.fit(X[it], y[it])
            rows.append({
                "split": split_name, "model": f"Random Forest + {feature_name}",
                **metrics(y[ix], rf.predict(X[ix])), "epochs": None
            })
            pred, epochs = fit_mlp(X, split)
            rows.append({
                "split": split_name, "model": f"MLP + {feature_name}",
                **metrics(y[ix], pred), "epochs": epochs
            })
        return rows

    results = (
        run_experiments("Random", random_split)
        + run_experiments("Scaffold", scaffold_split)
    )
    results_df = pd.DataFrame(results)
    mo.md(
        "## 4. Four model experiments\n\n"
        "The four model/feature combinations are trained separately for each "
        "split strategy. MLP feature and target scalers are fit on training "
        "data only; early stopping monitors validation MAE. Test data is used "
        "only for the final metrics below."
    )
    mo.ui.table(results_df.sort_values(["split", "MAE"]).reset_index(drop=True), selection=None)
    return results_df,


@app.cell
def _(mean_absolute_error, mean_squared_error, np, pd, random_split, results_df, scaffold_split, y):
    import marimo as mo

    baseline_rows = []
    for name, split in [("Random", random_split), ("Scaffold", scaffold_split)]:
        it, ix = split["train"], split["test"]
        pred = np.full(len(ix), float(np.mean(y[it])))
        baseline_rows.append({
            "split": name,
            "model": "Training-mean baseline",
            "MAE": float(mean_absolute_error(y[ix], pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y[ix], pred))),
            "R2": float(1 - np.sum((y[ix]-pred)**2) / np.sum((y[ix]-np.mean(y[ix]))**2)),
            "epochs": None,
        })
    final_table = pd.concat([results_df, pd.DataFrame(baseline_rows)], ignore_index=True)

    mo.md(r"""
    ## 5. Interpret the results

    - **MAE:** mean absolute error in log-solubility units; lower is better.
    - **RMSE:** penalizes larger errors more strongly; lower is better.
    - **R²:** variance explained; higher is better and can be negative.

    Compare each model with the training-mean baseline **within the same split
    strategy**. Do not interpret small differences from this one split as proof
    of universal superiority.

    ### Questions for your lab notes

    1. Do descriptors or Morgan fingerprints work better for each algorithm?
    2. Does the MLP outperform the Random Forest on either split?
    3. How much does test performance change between random and scaffold splits?
    4. Which compounds/chemotypes account for the largest errors?
    5. What additional validation would you need before making a strong claim?

    ### Limitations and takeaways

    ESOL is a small, single-endpoint benchmark. It does not establish performance
    on CYP inhibition, protein binding, or other endpoints. A rigorous benchmark
    would use repeated splits, a prespecified tuning protocol, and uncertainty
    estimates. This exercise is intended to teach a reproducible comparison,
    not to produce a publication-grade benchmark.

    **Deliverables:** save the result table, summarize the best combination
    within each split, and explain why scaffold splitting is more demanding
    than a random split.
    """)
    mo.ui.table(final_table.sort_values(["split", "MAE"]).reset_index(drop=True), selection=None)
    return


if __name__ == "__main__":
    app.run()
