import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader

    mo.md(
        """
        # Week 1 — PyTorch foundations

        **Project:** Generative AI for Life Sciences  
        **Assignment:** Build and evaluate a small neural network for a synthetic
        nonlinear regression task.

        This is a learning exercise. The generated data are **synthetic** and do
        not represent a real molecule, assay, or biological endpoint.
        """
    )
    return DataLoader, TensorDataset, mo, nn, np, plt, torch


@app.cell
def _(mo, np, torch):
    mo.md(
        """
        ## 1. Check the environment

        Run this cell and record the Python, PyTorch, NumPy, and device information.
        If PyTorch is not available in Molab, install it using the package manager
        panel rather than an IPython magic command.
        """
    )
    runtime_info = {
        "PyTorch": torch.__version__,
        "NumPy": np.__version__,
        "CUDA available": torch.cuda.is_available(),
        "Device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    runtime_info
    return


@app.cell
def _(mo, torch):
    mo.md(
        """
        ## 2. Tensors and autograd

        A tensor is PyTorch's multidimensional array. Autograd tracks operations
        on tensors and computes derivatives for tensors with `requires_grad=True`.

        **Exercise:** For `y = x² + 3x + 2`, calculate `dy/dx` at `x = 2`.
        Do this with autograd, then compare it with the analytical derivative
        `2x + 3`.
        """
    )
    # Create a tensor that tracks gradients
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 3 * x + 2
    # Compute gradients (backpropagation)
    y.backward()
    autograd_derivative = x.grad.item()
    analytical_derivative = 2 * 2 + 3
    #Check the gradient: dy/dx = 3 * x^2. Since x = 2.0, dy/dx = 3 * (2.0)^2 = 12.0
    derivative_check = {
        "y at x=2": y.item(), #extract a standard Python scalar value
        "autograd derivative": autograd_derivative,
        "analytical derivative": analytical_derivative,
    }
    derivative_check
    return


@app.cell
def _(mo, np, torch):
    mo.md(
        """
        ## 3. Create a synthetic regression dataset

        The target is a nonlinear function of one input feature, plus noise.
        We will split the data into training and validation sets before training.
        """
    )
    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = 600
    x_all = torch.linspace(-1.0, 1.0, n_samples).reshape(-1, 1) #changes it to a 2D tensor:
    noise = 0.12 * torch.randn(n_samples, 1)
    y_all = torch.sin(2 * torch.pi * x_all) + 0.3 * x_all + noise

    permutation = torch.randperm(n_samples)
    train_indices = permutation[:480]
    val_indices = permutation[480:]

    x_train, y_train = x_all[train_indices], y_all[train_indices]
    x_val, y_val = x_all[val_indices], y_all[val_indices]

    dataset_summary = {
        "Training rows": len(x_train),
        "Validation rows": len(x_val),
        "Input shape": tuple(x_train.shape),
        "Target shape": tuple(y_train.shape),
    }
    dataset_summary
    return x_train, x_val, y_train, y_val


@app.cell
def _(mo, plt, x_train, x_val, y_train, y_val):
    mo.md(
        """
        **Inspect the data before modeling.** Does the relationship look linear?
        What might a linear model fail to capture?
        """
    )
    fig_data, ax_data = plt.subplots()
    ax_data.scatter(x_train.numpy(), y_train.numpy(), s=12, alpha=0.45, label="train")
    ax_data.scatter(x_val.numpy(), y_val.numpy(), s=12, alpha=0.45, label="validation")
    ax_data.set(xlabel="Synthetic feature x", ylabel="Synthetic target y",
                title="Synthetic nonlinear regression data")
    ax_data.legend()
    fig_data
    return


@app.cell
def _(DataLoader, TensorDataset, mo, x_train, y_train):
    mo.md(
        """
        ## 4. Prepare batches

        `Dataset` groups inputs and targets. `DataLoader` yields mini-batches.
        A batch size of 32 is a reasonable starting point for this small example.
        """
    )
    batch_size = 32
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return (train_loader,)


@app.cell
def _(mo, nn):
    mo.md(
        """
        ## 5. Define the neural network

        The model maps one input feature to one continuous prediction. ReLU
        activations allow it to learn nonlinear relationships.

        **Exercise:** Before training, inspect the model architecture and explain
        what each `Linear` layer does.
        """
    )
    model = nn.Sequential(
        nn.Linear(1, 32),# projects the input into a 32-dimensional hidden space.
        nn.ReLU(),# introduces non-linearity, allowing the network to learn curves, bends, and complex patterns
        nn.Linear(32, 32),# takes the 32 features coming out of the first ReLU and transforms them into another 32 features.letting the model combine the patterns learned in the first layer into even more sophisticated representations.
        nn.ReLU(),
        nn.Linear(32, 1), # it takes the 32 hidden features and condenses them back down to 1 output value(prediction value)
    )
    model
    return (model,)


@app.cell
def _(mo, model, torch):
    mo.md(
        """
        ## 6. Configure loss and optimizer

        Mean squared error (MSE) penalizes squared prediction errors.
        Adam updates the model parameters using gradients.

        **TODO:** Change the learning rate from `0.01` to `0.003` in a second
        experiment later and compare the loss curves.
        """
    )
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    return loss_fn, optimizer


@app.cell
def _(loss_fn, mo, model, optimizer, train_loader):
    mo.md(
        """
        ## 7. Train the model

        Each epoch:
        1. set training mode;
        2. loop over mini-batches;
        3. clear old gradients;
        4. make predictions and calculate loss;
        5. backpropagate with `backward()`;
        6. update parameters with `optimizer.step()`.

        **TODO:** In your own words, explain why `optimizer.zero_grad()` is
        called before `loss.backward()`.
        If you do not clear the gradients before running backpropagation for the current      batch, the new gradients will mix with the old gradients from previous iterations.Accumulating old gradients causes the optimizer to take steps based on an incorrect, combined gradient rather than strictly the current mini-batch.  

        """
    )
    n_epochs = 1000
    training_losses = []

    for epoch in range(n_epochs):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            predictions = model(xb)
            loss = loss_fn(predictions, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        training_losses.append(sum(batch_losses) / len(batch_losses))

    training_losses[-1], n_epochs
    return (training_losses,)


@app.cell
def _(mo, plt, training_losses):
    mo.md(
        """
        **Inspect the learning curve.** A falling training loss suggests the
        model is fitting the training data. It does not, by itself, prove that
        the model generalizes to unseen data.
        """
    )
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(training_losses)
    ax_loss.set(xlabel="Epoch", ylabel="Training MSE", title="Training loss")
    fig_loss
    return


@app.cell
def _(mo, model, torch, x_val, y_val):
    mo.md(
        """
        ## 8. Evaluate on the validation set

        Use evaluation mode and disable gradient tracking during inference.
        Calculate MAE and RMSE. These are in the same units as the synthetic target.
        """
    )
    model.eval()
    with torch.no_grad():
        val_predictions = model(x_val)

    mae = torch.mean(torch.abs(val_predictions - y_val)).item()
    rmse = torch.sqrt(torch.mean((val_predictions - y_val) ** 2)).item()
    metrics = {"Validation MAE": mae, "Validation RMSE": rmse}
    metrics
    return (val_predictions,)


@app.cell
def _(mo, plt, val_predictions, x_val, y_val):
    mo.md(
        """
        ## 9. Compare predictions with targets

        A useful model should reproduce the overall nonlinear pattern, not merely
        memorize individual noisy observations.
        """
    )
    fig_pred, ax_pred = plt.subplots()
    ax_pred.scatter(x_val.numpy(), y_val.numpy(), s=16, alpha=0.5, label="true")
    ax_pred.scatter(x_val.numpy(), val_predictions.numpy(), s=16, alpha=0.5,
                    label="predicted")
    ax_pred.set(xlabel="Synthetic feature x", ylabel="Synthetic target y",
                title="Validation predictions")
    ax_pred.legend()
    fig_pred
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Your written interpretation

    Add your answers below (edit this cell or add a Markdown cell).

    1. What is a tensor, and how is its shape represented?
      like a multi-dimensional container or grid that holds numerical data.
      A tensor's shape is represented as a tuple of integers (specifically wrapped in a torch.Size object.torch.Size([3, 2]) :3 means there are 3 rows along the first dimension (axis 0). 2 means there are 2 elements in each row along the second dimension (axis 1).
    2. What did `requires_grad=True` do in the derivative exercise?
       to create and populate the .grad attribute on x
    3. Why do we call `optimizer.zero_grad()` during training?
       to clear out any old gradients left over from the previous iteration
    4. What is the difference between training loss and validation MAE/RMSE?
       Training loss is the internal mathematical scorecard the model uses to learn, while  Validation MAE/RMSE are human-readable scorecards used to evaluate how well the model actually performs on unseen data.
    5. Did the MLP capture the nonlinear pattern? What evidence supports your answer?
       -Visual Evidence (Predictions vs. Ground Truth Plot)
       -Low Validation Error (MAE or RMSE)
       -Training Loss Convergence
    6. Name one limitation of this experiment.
        poor extrapolation (the inability to generalize outside the training range).
    ## Week 1 extension tasks

    Complete at least **one** extension:

    - **Learning rate:** rerun with `lr=0.003`; compare the learning curves.
    - lr=0.01:
    - {
      "Validation MAE": 0.09970450401306152,
      "Validation RMSE": 0.12467148154973984
    }
    -lr=0.003🥇{
      "Validation MAE": 0.09970450401306152,
      "Validation RMSE": 0.12467148154973984
    }
    - **Model capacity:** change hidden widths from `32, 32` to `8, 8` or `64, 64`.
    - 8: MLP didn't capture the nonlinear pattern
    - 64: {
      "Validation MAE": 0.10414912551641464,
      "Validation RMSE": 0.1284957379102707
    }
    - **Linear baseline:** replace the MLP with one `nn.Linear(1, 1)` layer and compare validation MAE/RMSE.
    - **Checkpoint:** save `model.state_dict()` and load it into a newly created model.
     --- HOW TO SAVE ---
     Save just the state_dict using torch.save (standard file extension is .pth or .pt)
    torch.save(model.state_dict(), 'my_mlp_model.pth')


     --- HOW TO LOAD ---
     Step A: Create a brand-new instance of the EXACT same model architecture
    new_model = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

     Step B: Load the saved weights into the new model
    new_model.load_state_dict(torch.load('my_mlp_model.pth'))

     Step C: Put the model into evaluation mode (disables dropout/batchnorm updates if you use them)
    new_model.eval()

    ## Submission checklist

    - [ ] Notebook runs top to bottom without errors.
    - [ ] Autograd derivative matches the analytical derivative.
    - [ ] Validation MAE and RMSE are reported.
    - [ ] Prediction plot is included.
    - [ ] Written interpretation and one extension experiment are complete.
    - [ ] Commit the notebook and a 5–10 sentence summary to GitHub.
    """)
    return


if __name__ == "__main__":
    app.run()
