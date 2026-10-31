import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import seed_everything
from src.train import TrainConfig, fit, resolve_device, run_epoch


def test_training_updates_weights_and_saves_best(tmp_path):
    seed_everything(11)
    torch.set_num_threads(1)
    images = torch.randn(12, 3, 8, 8)
    targets = (images.mean(dim=(1, 2, 3)) > 0).long()
    loader = DataLoader(TensorDataset(images, targets), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(192, 2))
    before = model[1].weight.detach().clone()
    config = TrainConfig(epochs=2, device="cpu")
    history = fit(model, loader, loader, config, tmp_path / 'run', ['a', 'b'], 'fixture')
    assert not torch.equal(before, model[1].weight)
    assert len(history) == 2 and all(h['train_samples'] == 12 for h in history)
    state = torch.load(tmp_path / 'run/best.pt', weights_only=True)
    assert state['val_accuracy'] == max(h['val_accuracy'] for h in history)
    assert state['split_sha256'] == 'fixture'
    assert json.loads((tmp_path / 'run/config.json').read_text())['epochs'] == 2
    with pytest.raises(FileExistsError):
        fit(model, loader, loader, config, tmp_path / 'run', ['a', 'b'])


def test_validation_does_not_update_parameters():
    model = nn.Linear(3, 2)
    loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=2)
    before = model.weight.detach().clone()
    metrics = run_epoch(model, loader, torch.device('cpu'))
    assert torch.equal(before, model.weight) and metrics['samples'] == 4
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError):
        resolve_device('mps')
