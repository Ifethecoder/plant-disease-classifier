# Progress demo report template

Status: software implemented and CPU-tested; PlantVillage GPU results pending.
Replace placeholders only with evidence from a completed experiment directory.

## Research question and hypothesis

How does ImageNet pretraining affect plant disease classification performance when
limited labeled training data is available? Hypothesis: pretraining improves
validation performance at small labeled-data budgets. This is not yet a finding.

## Implemented protocol

Compare ResNet-18 random initialization with ImageNet-pretrained ResNet-18, fully
fine-tuned on the same class-aware, leaf-grouped nominal 10% subset. Adam lr 0.001,
batch size 32, five epochs, seed 42, 224×224 RGB with ImageNet normalization.
Select best validation accuracy, breaking ties by validation loss.

Real metadata audit: 43,596 published training and 10,709 test images. Quarantine
290 training records sharing upstream leaf identifiers with test. The development
split has 3,615 training and 8,646 validation images. The actual image fraction is
10.43% of the 34,660-image eligible training pool. Fallback identifiers limit the
physical-leaf disjointness guarantee. Test images are not evaluated.

## Completed experiment evidence (pending Colab)

- Experiment directory / source commit: [fill from saved environment]
- GPU, Python and package versions: [fill from environment.json]
- Dataset revision and split hash: [fill from splits.json]
- Random initialization validation accuracy / macro F1 / best epoch: [pending]
- ImageNet initialization validation accuracy / macro F1 / best epoch: [pending]
- Observed difference and interpretation: [pending; use comparison.csv]
- Learning curves: [attach learning_curves.png for both runs]
- Actual/predicted examples and mistakes: [attach figures and discuss errors]
- Inference demonstration: [show a validation image, true label and saved prediction]

## Implementation evidence

Offline unit/integration tests cover data splits/preprocessing, model shape and
weight selection, synthetic training, known-answer metrics, checkpoint reload,
artifact generation, inference, and notebook validity. The real metadata download
and official ImageNet weight loading have been checked separately. Software
verification is not plant disease classification performance evidence.

## Limitations and next steps

Run the real comparison in Colab. One seed, five epochs, class imbalance,
filename-fallback grouping, controlled-background images, shared optimization
settings, and checkpoint selection on validation constrain any conclusions.
Repeat at 1%, 3%, 10%, 30% and multiple seeds. Keep test untouched until final
model/protocol selection. Classifier placement experiments remain deferred.
