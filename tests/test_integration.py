"""Offline synthetic end-to-end checks; never research performance evidence."""
import csv
import json
import zipfile
from dataclasses import asdict, replace
from io import BytesIO

import nbformat
import pytest
import torch
from PIL import Image

from src.dataset import make_loaders, make_splits, materialize_development, seed_everything
from src.evaluate import comparison_table, evaluate_checkpoint
from src.models import build_model
from src.predict import Predictor
from src.train import TrainConfig, fit, run_epoch, save_json


def synthetic_metadata():
    classes = ['Apple___healthy', 'Tomato___healthy']
    rows = [{'image_path': f'raw/color/{classes[c]}/{g}_{photo}.jpg',
             'label': c, 'leaf_id': f'{c}-{g}'}
            for c in range(2) for g in range(8) for photo in range(2)]
    return {'classes': classes, 'revision': 'synthetic', 'records': {'train': rows, 'test': [
        {'image_path': 'raw/color/Apple___healthy/heldout.jpg', 'label': 0, 'leaf_id': 'test'}]}}


def test_archive_to_training_evaluation_and_inference(tmp_path, monkeypatch):
    seed_everything(42)
    torch.set_num_threads(1)
    metadata = synthetic_metadata()
    split = make_splits(metadata, fraction=.5)
    archive = tmp_path / 'synthetic.zip'
    image = Image.new('RGB', (64, 64), 'green')
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    with zipfile.ZipFile(archive, 'w') as z:
        for row in metadata['records']['train'] + metadata['records']['test']:
            z.writestr('enclosing_folder/' + row['image_path'], buffer.getvalue())
    monkeypatch.setattr('src.dataset.hf_hub_download', lambda *args, **kwargs: str(archive))
    development = materialize_development(metadata, split, tmp_path / 'cache')
    assert development.features['label'].names == metadata['classes']
    assert not list((tmp_path / 'cache/images').rglob('heldout.jpg'))
    config = TrainConfig(epochs=1, batch_size=4, image_size=64, device='cpu', fraction=.5)
    loaders = make_loaders(development, split, config.batch_size, image_size=config.image_size)
    duplicate = make_loaders(development, split, config.batch_size, image_size=config.image_size)
    assert list(iter(loaders[0].sampler)) == list(iter(duplicate[0].sampler))
    # Recreate loaders after inspecting sampler generators.
    loaders = make_loaders(development, split, config.batch_size, image_size=config.image_size)
    output = tmp_path / 'run'
    history = fit(build_model(2), *loaders, config, output, metadata['classes'], split['split_sha256'])
    metrics = evaluate_checkpoint(output / 'best.pt', loaders[1])
    assert metrics['accuracy'] == pytest.approx(history[0]['val_accuracy'])
    assert metrics['samples'] == len(loaders[1].dataset)
    assert metrics['split_sha256'] == split['split_sha256']
    assert (output / 'learning_curves.png').read_bytes().startswith(b'\x89PNG')
    assert (output / 'examples.png').exists()
    assert (output / 'misclassifications.png').exists() or (output / 'misclassifications.txt').exists()
    with (output / 'validation_predictions.csv').open() as f:
        predictions = list(csv.DictReader(f))
    assert len(predictions) == len(loaders[1].dataset)
    first = predictions[0]
    predictor = Predictor.from_checkpoint(output / 'best.pt')
    result = predictor.predict(tmp_path / 'cache/images' / first['image_path'])
    assert result['class_name'] == first['predicted']
    assert result['confidence'] == pytest.approx(float(first['confidence']), abs=1e-5)


def test_frozen_backbone_preserves_batchnorm_and_weights():
    torch.set_num_threads(1)
    model = build_model(2, freeze_backbone=True)
    before = model.bn1.running_mean.clone()
    weight = model.conv1.weight.detach().clone()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.randn(4, 3, 64, 64), torch.tensor([0, 1, 0, 1])), batch_size=2)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad))
    run_epoch(model, loader, torch.device('cpu'), optimizer)
    assert torch.equal(before, model.bn1.running_mean)
    assert torch.equal(weight, model.conv1.weight)
    assert model.fc.weight.grad is not None


def test_comparison_rejects_unequal_settings(tmp_path):
    runs = {}
    config = TrainConfig()
    for name, pretrained in (('random', False), ('pretrained', True)):
        output = tmp_path / name
        output.mkdir()
        save_json(output / 'config.json', asdict(replace(config, pretrained=pretrained)))
        save_json(output / 'validation_metrics.json', {'accuracy': .5, 'macro_f1': .4,
                  'split_sha256': 'synthetic', 'classes': ['a', 'b'], 'selected_epoch': 1, 'samples': 4})
        save_json(output / 'training_summary.json', {'seconds': 1})
        runs[name] = output
    assert len(comparison_table(runs, tmp_path)) == 2
    changed = asdict(replace(config, pretrained=True, learning_rate=.01))
    save_json(runs['pretrained'] / 'config.json', changed)
    with pytest.raises(ValueError, match='different training budgets'):
        comparison_table(runs, tmp_path)


def test_colab_notebook_valid_and_unexecuted():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'notebooks/progress_demo.ipynb'
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, 'notebook_cell', 'exec')


def test_matched_runner_shares_splits_and_budget(tmp_path, monkeypatch):
    from datasets import Dataset
    from src.experiment import run_comparison
    metadata = synthetic_metadata()
    metadata['upstream_schema'] = []
    metadata['published_counts'] = {k: len(v) for k, v in metadata['records'].items()}
    rows = [{**row, 'source_index': i, 'image': Image.new('RGB', (64, 64), 'green')}
            for i, row in enumerate(metadata['records']['train'])]
    development = Dataset.from_list(rows)
    monkeypatch.setattr('src.experiment.load_metadata', lambda *args: metadata)
    monkeypatch.setattr('src.experiment.materialize_development', lambda *args: development)
    initializations = []

    def offline_model(classes, pretrained, freeze):
        # Test orchestration offline. Real ImageNet loading was checked separately.
        initializations.append(pretrained)
        return build_model(classes, pretrained=False, freeze_backbone=freeze)

    monkeypatch.setattr('src.experiment.build_model', offline_model)
    config = TrainConfig(epochs=1, batch_size=4, fraction=.5, image_size=64, device='cpu')
    output = tmp_path / 'comparison'
    result = run_comparison(config, output, tmp_path / 'cache')
    assert initializations == [False, True]
    assert len(result) == 2 and (output / 'comparison.csv').exists()
    split_a = json.loads((output / 'random/splits.json').read_text())
    split_b = json.loads((output / 'pretrained/splits.json').read_text())
    assert split_a == split_b
    histories = [json.loads((output / name / 'history.json').read_text())
                 for name in ('random', 'pretrained')]
    assert histories[0][0]['train_samples'] == histories[1][0]['train_samples']
    assert histories[0][0]['val_samples'] == histories[1][0]['val_samples']
