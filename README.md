# Plant disease classification with transfer learning

A reproducible Minerva University Deep Learning progress demo addressing:

> How does ImageNet pretraining affect plant disease classification performance
> when limited labeled training data is available?

Compare **Model A**, a randomly initialized ResNet-18, with **Model B**, an
ImageNet-pretrained ResNet-18. Both use the same 38 output classes, architecture,
training subset, validation set, preprocessing, minibatch order and training budget.
The progress experiment uses approximately 10% of eligible training images.

**Current status:** implemented and CPU-tested. Real ImageNet weight loading and
live dataset metadata retrieval were verified. No GPU training or PlantVillage
classification experiment has been run. No research metrics are claimed. Run the
Colab notebook to produce the experimental results and figures.

## Quick start in Google Colab

The GitHub remote is `https://github.com/Ifethecoder/plant-disease-classifier.git`.
All implementation commits are local until the owner approves a push.

1. After publication, open [the progress notebook in Colab](https://colab.research.google.com/github/Ifethecoder/plant-disease-classifier/blob/main/notebooks/progress_demo.ipynb).
   Alternatively, choose **File → Open notebook → GitHub** in Colab and enter the
   repository URL. The link requires the notebook to be pushed and accessible.
2. Choose **Runtime → Change runtime type → GPU**.
3. Run the setup cell before importing torch. If libraries were already imported,
   restart the runtime after installation and run the notebook from the top.
4. Run all remaining cells. The notebook checks CUDA, mounts Google Drive by
   default, inspects the dataset, trains both models for five epochs, saves and
   displays validation metrics/plots, and demonstrates inference.
5. Keep the resulting Drive directory for your demo and progress report. If you
   disable Drive, download the optional results ZIP before ending the runtime.

Before pushing, you can upload the notebook and a ZIP of the project source into
Colab, extract the source, and set the setup cell's `PROJECT_DIR` to its root.
Do not include `.venv`, `.cache`, data or checkpoints in that source ZIP. Installation
requires internet. The automatic dataset download is a **2.18 GB archive**; allow
additional space for dependencies and extracted selected color images. Only
selected training/validation images are extracted. The archive includes other
configurations and test images, but no test image is opened or scored.

## Dataset, schema and leakage controls

Source: [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage),
**color** configuration, pinned revision
`9e97599868962bd0079b8db4b7f1efa9185fa1e7`.
The dataset card declares `image`, `image_path`, `label` (38 classes), `crop`,
`disease`, and `leaf_id`, with train/test splits. Actual published color manifests
contain **43,596 train** and **10,709 test** paths (54,305 total; use the manifests
rather than the card's rounded/descriptive total).

The repository uses a legacy Python dataset script; automatic conversion can
expose text rows instead of the documented image schema. Our adapter downloads
pinned metadata using `huggingface_hub`, reads the published manifests and leaf
map, then constructs a Hugging Face `datasets.Dataset` with `Image` and `ClassLabel`
features. It does not execute remote code or rely on deprecated script loading.
The adapter retains image path, label, physical leaf ID and original row index.
Crop/disease names are available through the class vocabulary.

The metadata audit was performed **before** implementing splits. It found **227
leaf IDs** shared between the published train/test lists under the upstream
loader's ID rules. The pipeline conservatively quarantines the **290 matching
training images**, preserving the entire test manifest. It then:

1. Partitions eligible training leaf groups within each class into approximately
   80% training and 20% validation, using seed 42 by default.
2. Selects a deterministic prefix of shuffled training leaf groups per class,
   using `ceil(fraction × number of groups)`, with at least one group per class.
3. Keeps all photographs within each selected leaf group together. The validation
   set is independent of the training fraction and model initialization.

Default counts verified from the real metadata:

| Nominal training group fraction | Training images | Fixed validation images |
|---|---:|---:|
| 1% | 467 | 8,646 |
| 3% | 1,193 | 8,646 |
| 10% | 3,615 | 8,646 |
| 30% | 10,530 | 8,646 |

The eligible training pool after validation/quarantine has 34,660 images. The 10%
group subset contains **10.43% of pool images**. Fractions are approximate in image
counts because whole leaf groups vary in size; per-class counts and the actual
image fraction are saved. Subsets are nested across fractions for a fixed seed.
See [the committed metadata audit](docs/dataset_audit.json) for schema, counts,
class coverage and hashes. Test metadata is used only for leakage checks; test
images and scores are reserved for final evaluation.

**Grouping limitation:** the upstream mapping falls back to filename-derived IDs
for 10,518 train and 2,676 test images. These groups are kept intact and overlapping
fallback IDs are quarantined conservatively. Unknown physical-leaf identities
cannot be proven disjoint; do not claim a complete physical-leaf leakage guarantee.

## Models and training protocol

| Setting | Model A | Model B |
|---|---|---|
| Architecture | ResNet-18, 38-way linear classifier | Same |
| Initialization | Random | `ResNet18_Weights.IMAGENET1K_V1` |
| Trainable layers | All | All (full fine-tuning) |
| Optimizer | Adam, lr 0.001, weight decay 0 | Same |
| Loss / batch / epochs | Cross-entropy / 32 / 5 | Same |
| Input | RGB, resize to 224×224 | Same |
| Normalization | ImageNet mean/std | Same |

No optimization settings differ by default. No augmentation is used in this
initial comparison, to keep the preprocessing controlled. Images use mean
`(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`. Python, NumPy, PyTorch,
CUDA and DataLoader seeds are set where practical. Deterministic PyTorch operations
are requested with warnings for unsupported cases; identical results across GPU
models, library versions and hardware are not guaranteed.

Both models see the same shuffled batches for the same number of epochs. A final
singleton training batch is dropped to avoid batch-normalization failures; the
same rule applies to both runs and the processed sample count is saved per epoch.
No validation sample is dropped. Best checkpoints maximize validation accuracy,
with validation loss as a tie-breaker. Optional `--freeze-backbone` freezes weights
and batch-normalization statistics for later experimentation; it is off for the
demo. Classifier placement experiments are deliberately deferred.

## Local setup and reproducible commands

Use Python 3.11 or 3.12 (tested on Python 3.12, macOS ARM64 CPU):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

`requirements.txt` pins direct dependencies. `requirements-lock.txt` records the
full tested Python 3.12 environment; it can be used for a closer reproduction on
compatible systems. GPU runtime details and installed versions are also saved
per experiment. Google Colab supplies its own notebook environment.

Run both models with a single shared data preparation step:

```bash
python -m src.experiment --fraction 0.10 --epochs 5 --batch-size 32 \
  --learning-rate 0.001 --seed 42 --device cuda \
  --output results/demo_seed42 --cache-dir /tmp/plant-village-cache
```

Use `--device cpu` for local training; full experiments can be slow. For independent
runs with the same deterministic split:

```bash
python -m src.train --output results/random_seed42 --device cuda
python -m src.train --pretrained --output results/pretrained_seed42 --device cuda
```

Both CLIs also accept `--val-fraction`, `--image-size`, `--num-workers`, and
`--freeze-backbone`. `--help` lists all options. Use new output directories to
avoid overwriting previous runs. In the comparison CLI initialization is selected
automatically for the two runs. The single-model CLI saves training artifacts;
the comparison CLI additionally generates evaluation artifacts and a table.

For evaluation from Python, reconstruct the development data/loaders with the
saved configuration using `src.dataset` and call
`src.evaluate.evaluate_checkpoint(checkpoint, validation_loader)`. Use the
sequential validation loader returned by `make_loaders`; shuffled or dropping
loaders are rejected. The Colab notebook does this automatically via the runner.

## Evaluation and artifacts

Accuracy counts all correct predictions. Macro F1 averages per-class F1 equally
across the full class vocabulary, with zero division set to zero. Each run saves a
confusion matrix, raw prediction CSV and aggregate metrics. Validation is used
both for checkpoint selection and progress reporting; its scores are development
estimates. No test evaluation entry point is included in this progress demo.

```text
results/<experiment>/
  splits.json                  # indices, classes, counts, quarantine, hashes
  dataset_schema.json
  comparison.csv
  comparison.md
  random/                      # also pretrained/
    config.json
    environment.json           # versions, source commit, GPU, split identity
    splits.json
    history.json
    training_summary.json
    best.pt
    validation_metrics.json
    validation_predictions.csv
    learning_curves.png         # loss and accuracy
    examples.png                # actual/predicted labels and confidence
    misclassifications.png      # or a note when none exist
```

Configuration, split manifest and environment are saved before training starts;
history and best checkpoint are updated after each epoch. Interrupted runs retain
completed-epoch evidence but cannot be resumed automatically. Use a new output
folder to rerun. Image writes and checkpoint/history updates use temporary files.
Datasets, cached weights, checkpoints and generated results are ignored by Git.
Persist the complete experiment directory to Drive or download it separately.

## Inference

```python
from src.predict import Predictor
predictor = Predictor.from_checkpoint("results/demo_seed42/pretrained/best.pt")
result = predictor.predict("path/to/leaf.jpg")  # also accepts PIL.Image
print(result)  # class_index, class_name, crop, disease, confidence
```

Or use `python -m src.predict --checkpoint PATH --image PATH`. The checkpoint stores
the class mapping and image size; inference rebuilds the architecture without
redownloading pretrained weights. Confidence is a softmax score, not a calibrated
probability. The notebook demonstrates this on a validation image with a known
actual label. There is no frontend.

## Layout and evidence

`src/dataset.py`, `models.py`, `train.py`, `evaluate.py`, and `predict.py` implement
the core functionality. The extra `src/experiment.py` coordinates the matched
comparison without duplicating notebook training code. `tests/` contains offline
unit/integration checks; `docs/` contains the real metadata audit, verification
record, and a progress-report template. `notebooks/progress_demo.ipynb` orchestrates
the Colab demo. `results/` holds generated artifacts outside version control.

See [verification evidence](docs/verification.md) and the
[progress-report template](docs/progress_report.md). Synthetic tests verify code,
not disease recognition. GPU runs and the real image archive integration remain
unexecuted locally. The notebook has been schema-validated and its code cells
compiled, but it has not been executed in Colab.

## Limitations and next experiments

Five epochs, one seed and a single shared learning rate are preliminary: they may
favor one initialization and do not establish each method's best attainable
performance. PlantVillage has controlled image backgrounds and class imbalance;
performance may not transfer to real field photographs. The grouping fallback
and validation-based checkpoint selection further limit conclusions. Results are
not yet available and should not be filled in until training completes.

Next, run 1%, 3%, 10% and 30% training group fractions with multiple fixed seeds,
report variation across seeds, and hold the validation set fixed across fractions
within each seed. Consider optimization sensitivity after the controlled initial
comparison. Reserve the published test set for the final, frozen evaluation
protocol. Classifier placement experiments are planned for the final project.

## Dataset attribution

Mohanty, S. P., Hughes, D. P., and Salathé, M. (2016).
[Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419).
Dataset hosted by [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage);
its card declares CC BY-SA 3.0. Cite the dataset/paper in your report and retain
appropriate attribution if redistributing data or image examples. Source split
and leaf-map logic were inspected in the pinned upstream loader.
