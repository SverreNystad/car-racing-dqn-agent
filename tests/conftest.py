import pytest

import wandb


@pytest.fixture(autouse=True)
def init_wandb():
    # Initialize wandb in disabled mode to avoid actual logging during tests
    wandb.init(project="test_project", mode="disabled")
    yield
    wandb.finish()
