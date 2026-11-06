# Generative AI for Life Sciences

A hands-on portfolio project for building practical deep-learning and generative-AI skills for scientific applications.

**Working environment:** [marimo Molab](https://molab.marimo.io/)  
**Notebook format:** marimo `.py` notebooks (Git-friendly, executable Python)  
**Weekly commitment:** approximately 10–15 hours

## Project goals

This repository will grow alongside a 12-week learning plan. The goal is to demonstrate that I can:

- build and evaluate deep-learning models with PyTorch;
- work with molecular and biomedical data;
- use transformer models and scientific language models;
- build a literature-search / retrieval-augmented generation (RAG) prototype;
- evaluate model outputs and communicate scientific limitations clearly.

The first assignment uses a **synthetic regression dataset**. It is deliberately small and does not represent experimental measurements or a real biological endpoint. The purpose is to learn the PyTorch workflow before introducing molecular datasets.

## Repository layout

```text
generative-ai-life-sciences/
├── README.md
├── pyproject.toml
├── .gitignore
├── notebooks/
│   └── week01_pytorch_foundations.py
├── src/
│   └── life_sciences_ai/
│       └── __init__.py
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── reports/
│   └── .gitkeep
└── tests/
    └── test_environment.py
```

### Folder conventions

- `notebooks/`: exploratory work and weekly marimo assignments.
- `src/life_sciences_ai/`: reusable Python functions once notebook code is stable.
- `data/raw/`: original datasets; do not commit private, restricted, or large datasets.
- `data/processed/`: cleaned or transformed data; generally keep generated data out of Git unless it is small and permitted.
- `reports/`: figures, short write-ups, and evaluation summaries.
- `tests/`: lightweight checks for reusable code and environment assumptions.

## Start in marimo Molab

Molab is a cloud-hosted marimo notebook environment. It includes common scientific packages, including PyTorch and NumPy, and its package manager can install missing imports. Molab notebooks are public but not discoverable by default; do not put secrets, private data, or unpublished confidential work in a notebook.

1. Create a GitHub repository named `generative-ai-life-sciences` (public is useful for a portfolio; choose private if you prefer to work privately first).
2. Upload or commit this starter folder to the repository.
3. Open [Molab](https://molab.marimo.io/).
4. Use Molab's GitHub integration / mirror option to open the repository as the source of truth. Select `notebooks/week01_pytorch_foundations.py`.
5. Run the notebook from top to bottom. If a package is missing, use Molab's package manager panel to install it; do not use IPython `%pip` or `%conda` magics.
6. Save or sync your changes back to GitHub using the repository/mirror workflow shown in Molab.
7. Check the GitHub repository after syncing to confirm the notebook and any README changes are present.

If the GitHub mirror workflow is not available in your current Molab session, use the notebook's download option, then upload the updated `.py` file to GitHub through the repository's web interface.

Useful documentation:
- [Molab guide](https://docs.marimo.io/guides/molab/)
- [marimo package management](https://docs.marimo.io/guides/package_management/)
- [marimo notebooks and quickstart](https://docs.marimo.io/)

## Python environment

The `pyproject.toml` defines a reproducible local environment for later use outside Molab.

Recommended Python: **3.11 or newer**. Molab manages its cloud runtime separately, so you do not need to create a virtual environment inside Molab.

### Local setup (optional)

Install `uv`, then from the repository root:

```bash
uv sync
uv run marimo edit notebooks/week01_pytorch_foundations.py
```

If you prefer `pip` and a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
marimo edit notebooks/week01_pytorch_foundations.py
```

PyTorch installation can vary by operating system and CPU/GPU setup. If local installation fails, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) rather than guessing a CUDA build.

## Week 1 — PyTorch foundations

**Estimated time:** 10–12 hours

### Learning objectives

By the end of Week 1, I should be able to:

1. explain the difference between a Python number, a NumPy array, and a PyTorch tensor;
2. create tensors and inspect their shape, dtype, and device;
3. explain how `requires_grad`, `backward()`, and `.grad` work;
4. define a small neural network with `torch.nn`;
5. train a model using a loss function, optimizer, and training loop;
6. evaluate regression predictions using MAE and RMSE;
7. explain why a validation set is separate from training data.

### Lessons

- [fast.ai — Practical Deep Learning for Coders, Lesson 1](https://course.fast.ai/Lessons/lesson1.html): focus on the practical workflow and the idea of learning from examples.
- [PyTorch — Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html): focus on tensors, datasets/data loaders, building a model, autograd, optimization, and saving/loading models.

Do not try to master every detail in one week. Prioritize understanding and modifying the training loop.

### Assignment

Open `notebooks/week01_pytorch_foundations.py` in Molab and complete the TODOs.

**Core tasks**

- Inspect the runtime and tensor properties.
- Calculate a derivative with autograd and explain the result.
- Train an MLP on the supplied synthetic nonlinear regression problem.
- Plot training loss and predictions versus targets.
- Calculate validation MAE and RMSE.
- Write a short interpretation of whether the model learned the nonlinear relationship.

**Stretch tasks**

- Change the hidden-layer width and compare results.
- Change the learning rate and describe what happens.
- Compare the MLP with a simple linear model.
- Save the trained model's `state_dict` and reload it for inference.

### GitHub deliverables

Commit these items:

- completed `notebooks/week01_pytorch_foundations.py`;
- a short `reports/week01_summary.md` (about 5–10 sentences);
- at least one plot showing predictions versus true values.

Do not commit large model checkpoints or raw datasets.

### Definition of done

- [ ] Notebook runs from the first cell to the last without errors.
- [ ] I can explain tensors, autograd, `nn.Module` / `nn.Sequential`, loss, optimizer, and validation data in my own words.
- [ ] The notebook reports validation MAE and RMSE.
- [ ] The plot and short written interpretation are committed to GitHub.
- [ ] I can describe one experiment I changed and what effect it had.

## Weekly work rhythm

A realistic 12-hour week:

| Activity | Hours |
|---|---:|
| Course lessons and notes | 3 |
| Coding and experiments | 5 |
| Reading one relevant paper or technical guide | 1 |
| GitHub cleanup and write-up | 1 |
| Job applications and networking | 2 |

Keep job searching active during the learning plan. A reasonable baseline is two or three targeted applications per week, plus one networking follow-up.

## Scientific integrity and portfolio standards

- Label synthetic data as synthetic.
- Distinguish measured experimental values from simulated, predicted, or generated values.
- Report the split strategy and metrics; do not report training performance as generalization performance.
- Record package versions and important settings.
- Include limitations and failure cases in every project README.
- Never upload confidential employer data, private patient data, API keys, or data you are not licensed to redistribute.
