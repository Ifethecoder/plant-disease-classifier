import pytest

from src.evaluate import classification_metrics, plot_history


def test_known_metrics_and_missing_classes():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert metrics['accuracy'] == 0.75
    assert metrics['macro_f1'] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics['confusion_matrix'] == [[1, 1], [0, 2]]
    assert classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 3)['macro_f1'] == pytest.approx((2/3 + .8)/3)
    for targets, preds in (([], []), ([0], [2]), ([0, 1], [1])):
        with pytest.raises(ValueError):
            classification_metrics(targets, preds, 2)


def test_history_plot(tmp_path):
    history = [{'epoch': 1, 'train_loss': 1., 'val_loss': 1.2,
                'train_accuracy': .5, 'val_accuracy': .4}]
    plot_history(history, tmp_path / 'curves.png')
    assert (tmp_path / 'curves.png').stat().st_size > 100
