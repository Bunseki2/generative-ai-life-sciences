# Revised Week 2 — ESOL molecular ML comparison

Open `week02_esol_four_model_comparison.py` with marimo.

The notebook downloads the ESOL/Delaney CSV, computes 12 RDKit 2D descriptors
and 2048-bit Morgan fingerprints (radius 2), then compares:
- Random Forest + descriptors
- MLP + descriptors
- Random Forest + Morgan fingerprints
- MLP + Morgan fingerprints

All four experiments run under both random and Bemis–Murcko scaffold splits.
A training-mean baseline is included. MLP scalers are fitted on training data
only, and early stopping uses validation data only.

Install dependencies in Molab if needed:

```bash
uv pip install pandas numpy scikit-learn torch rdkit marimo
```

Dataset source: DeepChem Delaney dataset
https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv

Background: https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html

This is a learning exercise, not a publication-grade benchmark. Random and
scaffold test results represent different generalization settings.
