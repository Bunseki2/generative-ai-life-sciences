# Generative AI for Life Sciences — Weeks 2–12

This supplement contains marimo `.py` notebooks for Weeks 2–12 of the 12-week learning plan. Merge the `notebooks/` folder into your existing `generative-ai-life-sciences` GitHub repository, keeping your completed Week 1 notebook.

## How to use in Molab

1. Upload/merge these notebooks into your existing GitHub repository.
2. Open the notebook for the current week in [marimo Molab](https://molab.marimo.io/).
3. Install only the packages needed for that week's work through Molab's package manager.
4. Run from top to bottom, complete TODOs, add Markdown explanations, and save/sync to GitHub.
5. Commit a short `reports/weekNN_summary.md` for each week.

## Notebook sequence

| Week | Notebook topic | Main output |
|---|---|---|
| 2 | Molecular data and classical ML baseline | RDKit descriptors/fingerprints and regression baseline |
| 3 | Transformer foundations and biomedical text | Tokenization and attention-mask inspection |
| 4 | Transformer inference and embeddings | PubMedBERT embeddings and similarity |
| 5 | Biomedical text classification and fine-tuning | Fine-tuning pipeline and error analysis |
| 6 | Datasets, embeddings, and semantic search | FAISS semantic retrieval and Recall@k |
| 7 | Instruction tuning and LoRA | Reviewed instruction dataset and adapter experiment |
| 8 | LLM evaluation and error analysis | Evaluation rubric and scored test set |
| 9 | Biomedical literature RAG assistant | Evidence-grounded retrieval and cited answers |
| 10 | Protein language models with ESM-2 | Protein embeddings and sequence analysis |
| 11 | Scientific AI assistant: CYP evidence prototype | Structured CYP evidence with source types |
| 12 | Portfolio polish and technical communication | Reproducible projects and interview stories |

## Important notes

- Several notebooks deliberately use small illustrative examples to demonstrate APIs. These are **not** sufficient for scientific claims or a job-portfolio result.
- For Weeks 5, 7, and 9, choose model sizes appropriate to Molab's memory and compute limits. Fine-tuning may require a GPU; start with a small model or CPU-compatible dry run.
- For Week 5, replace toy labels with a larger, licensed dataset before reporting performance.
- For Week 9, use PubMed/Europe PMC or another source according to its API terms and reuse license. Keep PMID/DOI and source URLs attached to each chunk.
- For Week 10, sequence similarity or embedding similarity does not prove shared function, structure, or binding.
- Week 11 should build on your existing OpenADMET CYP Challenge/CheMeleon work, but clearly distinguish measured assay values from model predictions.
- Keep data files, checkpoints, and credentials out of Git unless redistribution is permitted and the files are appropriately small.

## Notebook implementation note

Each weekly exercise is organized as a self-contained main marimo cell so that
variables remain in scope and the assignment can run sequentially in Molab.
You can split the main cell into smaller cells later by explicitly returning
variables from one cell and accepting them as arguments in the next.
