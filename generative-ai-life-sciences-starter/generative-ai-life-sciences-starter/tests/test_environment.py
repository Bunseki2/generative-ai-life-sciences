def test_project_package_imports():
    import life_sciences_ai

    assert life_sciences_ai.__version__ == "0.1.0"


def test_torch_tensor_operations():
    import torch

    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.shape == (3,)
    assert torch.isclose(x.sum(), torch.tensor(6.0))
