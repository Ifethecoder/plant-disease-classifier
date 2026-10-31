import pytest
import torch
from PIL import Image
from torch import nn

from src.predict import predict_image


def test_inference_format_and_path_support(tmp_path):
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.copy_(torch.tensor([0., 2.]))
    image = Image.new('RGB', (64, 64), 'green')
    path = tmp_path / 'leaf.png'
    image.save(path)
    classes = ['Apple___healthy', 'Tomato___Early_blight']
    result = predict_image(image, model, classes)
    assert result == predict_image(path, model, classes)
    assert result['class_index'] == 1
    assert result['class_name'] == 'Tomato___Early_blight'
    assert result['crop'] == 'Tomato' and result['disease'] == 'Early_blight'
    assert result['confidence'] == pytest.approx(torch.softmax(torch.tensor([0., 2.]), 0)[1].item())
    assert 0 <= result['confidence'] <= 1
    with pytest.raises(TypeError):
        predict_image(123, model, classes)
