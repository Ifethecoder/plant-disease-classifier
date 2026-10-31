from unittest.mock import patch

import pytest
import torch
from torchvision.models import ResNet18_Weights, resnet18

from src.models import build_model


def test_random_model_output_and_full_training():
    torch.set_num_threads(1)
    model = build_model(38)
    model.eval()
    with torch.inference_mode():
        assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 38)
    assert all(p.requires_grad for p in model.parameters())


def test_pretrained_requests_correct_weights_and_optional_freeze():
    # Unit test avoids network; a separate real weight-loading check is documented.
    with patch('src.models.resnet18', return_value=resnet18(weights=None)) as factory:
        model = build_model(38, pretrained=True, freeze_backbone=True)
    factory.assert_called_once_with(weights=ResNet18_Weights.IMAGENET1K_V1)
    assert model.fc.out_features == 38
    assert model.fc.weight.requires_grad
    assert not model.layer1[0].conv1.weight.requires_grad
    with pytest.raises(ValueError):
        build_model(1)
