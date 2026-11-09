import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # Week 07 — Instruction tuning and LoRA

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
    from transformers import AutoTokenizer

    # Inspect the chat template first. The model is intentionally small for a first run.
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    messages = [
        {"role": "system", "content": "You are a careful scientific assistant. Distinguish evidence from hypotheses."},
        {"role": "user", "content": "What is the difference between an assay result and a model prediction?"},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(rendered)

    examples = [
        {"instruction": "Define molecular dynamics simulation.",
         "response": "A computational method that simulates time-dependent atomic motion using a physical model."},
        {"instruction": "Does a model prediction prove binding?",
         "response": "No. A prediction is a hypothesis and requires appropriate experimental validation."},
        {"instruction": "What is a SMILES string?",
         "response": "A text representation of a molecular structure."},
    ]
    print("Initial reviewed examples:", len(examples))

    # TODO:
    # 1. Expand to 30-100 carefully reviewed instruction/response examples.
    # 2. Split into train and validation sets.
    # 3. Load AutoModelForCausalLM and configure LoRA with PEFT.
    # 4. Select target_modules for the chosen model architecture; do not copy blindly.
    # 5. Train with TRL SFTTrainer if hardware allows; otherwise document a dry run.
    # 6. Save only the adapter, then compare base and adapted model on fixed prompts.
    # Explain full fine-tuning vs LoRA, adapter weights, and chat-template consistency.
    return


@app.cell
def _():
    import marimo as mo
    mo.md("""
    ## GitHub deliverable

    Complete the TODOs, add your own explanations and plots, and save a short
    `reports/week07_summary.md`.

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
