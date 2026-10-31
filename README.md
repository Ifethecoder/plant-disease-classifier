# Plant disease classification

In this Deep Learning project, I investigate **how ImageNet pretraining affects
plant disease classification when labeled training data is limited**. The comparison
uses ResNet-18 with random weights and ResNet-18 with ImageNet-pretrained weights.
Both models predict the same 38 PlantVillage crop/disease classes.

The implementation passes CPU tests, including synthetic training and inference.
The PlantVillage GPU experiment has not yet been run, so no classification results
are reported here. The notebook has no saved training outputs.

## Experiment

The dataset consists of color images from [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage),
pinned to revision `9e97599868962bd0079b8db4b7f1efa9185fa1e7`.
The repository uses a legacy loading script. The published manifests and leaf map
provide the records for a Hugging Face image dataset without executing remote code.

The published test set stays intact. Eligible training data is split by leaf within
each class, keeping photographs with the same known leaf identifier together.
The audit found overlapping identifiers in the published splits, leading to the
exclusion of 290 matching training images. Filename fallbacks prevent a complete
guarantee of separation for unidentified physical leaves. Counts and grouping
limitations are recorded in [dataset_audit.md](docs/dataset_audit.md).

For seed 42, the nominal 10% subset contains 3,615 images and the fixed validation
set contains 8,646. Sampling whole leaf groups within each class gives an actual
image fraction of 10.43% of the eligible training pool. Changing the training
fraction does not change validation, and the smaller training subsets are nested.

| Setting | Both models |
|---|---|
| Architecture | ResNet-18 with a new 38-class final layer |
| Trainable layers | All layers |
| Optimizer and loss | Adam, lr 0.001, cross-entropy |
| Batch size and epochs | 32 and 5 |
| Input | RGB, resized to 224×224 |
| Normalization | ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |

Both models use identical preprocessing, shuffle seeds and optimization settings,
without augmentation in this initial comparison. Checkpoint selection uses the
highest validation accuracy, breaking ties by validation loss. Evaluation reports
accuracy and macro F1, averaging F1 equally across all 38 classes. Validation scores
are development estimates because the same set also supports checkpoint selection.
Test images remain unopened and unevaluated.

## Running the demo in Colab

1. **Notebook:** after publication, [progress_demo.ipynb opens in Colab](https://colab.research.google.com/github/Ifethecoder/plant-disease-classifier/blob/main/notebooks/progress_demo.ipynb).
   Another route is **File → Open notebook → GitHub**, using `Ifethecoder/plant-disease-classifier`.
2. **Runtime:** **Runtime → Change runtime type → GPU**, followed by the cells in order.
3. **Dependencies:** installation comes before importing PyTorch. If packages were
   already imported, a runtime restart and another run from the top are necessary.
4. **Storage:** `SAVE_TO_DRIVE = True` preserves checkpoints, losses, metrics and
   figures. With Drive disabled, results need downloading before disconnecting.

The GitHub changes are still local. Before publication, the notebook and a source
ZIP can be uploaded into Colab. After extraction, `PROJECT_DIR` in the first cell
points to the project root. The source ZIP excludes `.venv`, `.cache`, data and
checkpoints.

The first image download is a 2.18 GB archive, with extra space needed for extracted
images and dependencies. The archive is cached outside Git; only selected training
and validation color images are extracted. Each run gets a new output directory.
Completed epoch histories and checkpoints are saved, but automatic resume is not
implemented.

## Following the code

| File | Purpose |
|---|---|
| `src/dataset.py` | Metadata, leaf-based splits, image loading, transforms and batches |
| `src/models.py` | ResNet-18 construction, checkpoint loading and single-image inference |
| `src/train.py` | Training/validation loops and best-checkpoint saving |
| `src/evaluate.py` | Accuracy, macro F1, learning curves and prediction figures |
| `notebooks/progress_demo.ipynb` | Two-model experiment, results table and inference demo |
| `tests/test_project.py` | Small offline checks using synthetic data |

The notebook loads metadata, saves fixed split indices, downloads the selected
images, and loops over the two initializations. Each run starts with the same seed
and fresh loaders, then trains, reloads the best checkpoint, and evaluates on the
same validation set. The final cells display the comparison, curves, examples and
mistakes.

The run directory contains shared `settings.json` and `splits.json`, a comparison
CSV/Markdown table, and a folder per model with its settings, epoch history, best
checkpoint, metrics, prediction CSV and figures. Checkpoints store class names and
image size for inference. `predict_image` accepts a PIL image or path and returns
the crop/disease class and softmax confidence. Confidence is not calibrated.

## Local checks and reproducibility

The project supports Python 3.11 or 3.12. Local checks used Python 3.12, PyTorch 2.6.0
and torchvision 0.21.0 on a macOS ARM64 CPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Direct dependencies are pinned in `requirements.txt`; `requirements-lock.txt`
records the complete tested environment. The notebook saves settings, package
versions, dataset revision and split indices. Python, NumPy and
PyTorch are seeded, with deterministic operations requested where practical.
Results can still vary across hardware. Both runs drop a final singleton training
batch to avoid batch-normalization failures; validation never drops samples.

All **8 tests pass**. They cover leaf grouping and nested subsets, preprocessing,
model outputs/weight selection, known metrics, synthetic training and checkpoint
reload, inference, and notebook validity. The notebook's comparison cell also ran
on synthetic images with real random and cached ImageNet models on CPU. Matplotlib
emits dependency deprecation warnings, but the plots pass their checks. These checks
are software verification, not plant disease classification results. The real
image-archive download and full GPU experiment still need a Colab run.

## Next steps and limitations

I plan to repeat the comparison at 1%, 3%, 10% and 30% with multiple seeds, keeping
validation fixed across fractions within each seed. Optional backbone freezing is
available in `build_model`, but this demo uses full fine-tuning. Classifier
placement experiments are deferred.

A short run and one seed cannot establish the best attainable performance of each
initialization. Shared learning rates may favor one model. Class imbalance,
filename-fallback grouping and PlantVillage's controlled backgrounds also limit
the conclusions. Test evaluation is reserved for the final experimental protocol.

Dataset reference: Mohanty, Hughes and Salathé (2016),
[Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419).
The Hugging Face dataset card declares CC BY-SA 3.0; shared dataset images retain
appropriate attribution.
