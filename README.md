# Plant disease classification

In this Deep Learning project, I investigate **how ImageNet pretraining affects
plant disease classification when labeled training data is limited**. I compare
ResNet-18 with random weights against ResNet-18 with ImageNet-pretrained weights.
Both models predict the same 38 PlantVillage crop/disease classes.

The implementation passes CPU tests, including synthetic training and inference.
I have not yet run the PlantVillage experiment on a GPU, so I do not report
classification results here. The notebook has no saved training outputs.

## My experiment

I use the color images from [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage),
pinned to revision `9e97599868962bd0079b8db4b7f1efa9185fa1e7`.
The repository uses a legacy loading script, so I download its published manifests
and leaf map, then build a Hugging Face image dataset without executing remote code.

I preserve the published test set and split the eligible training data by leaf
within each class. Photographs with the same known leaf identifier stay together.
My audit found overlapping identifiers in the published splits; I exclude 290
matching training images. Some identifiers use filename fallbacks, so I cannot
guarantee separation for every unidentified physical leaf. My counts and grouping
limitations are recorded in [dataset_audit.md](docs/dataset_audit.md).

For seed 42, my nominal 10% subset contains 3,615 images and my fixed validation set
contains 8,646. I sample whole leaf groups within each class, so the actual image
fraction is 10.43% of the eligible training pool. Changing the training fraction
does not change validation, and the smaller training subsets are nested.

| Setting | Both models |
|---|---|
| Architecture | ResNet-18 with a new 38-class final layer |
| Trainable layers | All layers |
| Optimizer and loss | Adam, lr 0.001, cross-entropy |
| Batch size and epochs | 32 and 5 |
| Input | RGB, resized to 224×224 |
| Normalization | ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |

I use identical preprocessing, shuffle seeds and optimization settings. I do not
use augmentation in this initial comparison. I select each best checkpoint by
validation accuracy, breaking ties by validation loss. I report accuracy and macro
F1, averaging F1 equally across all 38 classes. Validation scores are development
estimates because I also use them for checkpoint selection. Test images remain
unopened and unevaluated.

## Running my demo in Colab

1. After publishing the repository, I open [progress_demo.ipynb in Colab](https://colab.research.google.com/github/Ifethecoder/plant-disease-classifier/blob/main/notebooks/progress_demo.ipynb),
   or use **File → Open notebook → GitHub** with `Ifethecoder/plant-disease-classifier`.
2. I select **Runtime → Change runtime type → GPU** and run the cells in order.
3. I install dependencies before importing PyTorch. If packages were already
   imported, I restart the runtime after installation and rerun the notebook.
4. I keep `SAVE_TO_DRIVE = True` to persist checkpoints, losses, metrics and figures.
   With Drive disabled, I download results before disconnecting.

The GitHub changes are still local. Before publication, I can upload the notebook
and a source ZIP into Colab, extract the project, and set `PROJECT_DIR` in the first
cell to its root. I exclude `.venv`, `.cache`, data and checkpoints from that ZIP.

The first image download is a 2.18 GB archive; I allow extra space for extracted
images and dependencies. I cache it outside Git and extract only my selected
training/validation color images. I start a new output directory for each run.
Completed epoch histories and checkpoints are saved, but automatic resume is not
implemented.

## Following the code

| File | Purpose |
|---|---|
| `src/dataset.py` | Metadata, leaf-based splits, image loading, transforms and batches |
| `src/models.py` | ResNet-18 construction, checkpoint loading and single-image inference |
| `src/train.py` | Training/validation loops and best-checkpoint saving |
| `src/evaluate.py` | Accuracy, macro F1, learning curves and prediction figures |
| `notebooks/progress_demo.ipynb` | My explicit two-model experiment, results table and inference demo |
| `tests/test_project.py` | Small offline checks using synthetic data |

In the notebook I load metadata, save fixed split indices, download the selected
images, and loop over the two initializations. For each model I reset the seed,
recreate the loaders, train, reload the best checkpoint, and evaluate on the same
validation set. I then display the comparison, curves, examples and mistakes.

My run directory contains shared `settings.json` and `splits.json`, a comparison
CSV/Markdown table, and a folder per model with its settings, epoch history, best
checkpoint, metrics, prediction CSV and figures. Checkpoints store class names and
image size for inference. I call `predict_image` with a PIL image or path; it returns
the crop/disease class and softmax confidence. Confidence is not calibrated.

## Local checks and reproducibility

I use Python 3.11 or 3.12. My local check environment uses Python 3.12, PyTorch 2.6.0
and torchvision 0.21.0 on a macOS ARM64 CPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

I pin direct dependencies in `requirements.txt`; `requirements-lock.txt` records
the complete tested environment. The notebook saves settings, package versions,
source commit, dataset revision and split indices. I seed Python, NumPy and
PyTorch and request deterministic operations where practical. Results can still
vary across hardware. Both runs drop a final singleton training batch to avoid
batch-normalization failures; validation never drops samples.

All **8 tests pass**. They cover leaf grouping and nested subsets, preprocessing,
model outputs/weight selection, known metrics, synthetic training and checkpoint
reload, inference, and notebook validity. I also ran the notebook's comparison
cell on synthetic images with real random and cached ImageNet models on CPU.
Matplotlib emits dependency deprecation warnings, but the plots pass their checks.
These checks are software verification, not plant disease classification results.
The real image-archive download and full GPU experiment still need a Colab run.

## Next steps and limitations

I plan to repeat the comparison at 1%, 3%, 10% and 30% with multiple seeds, keeping
validation fixed across fractions within each seed. I retain optional backbone
freezing in `build_model`, but use full fine-tuning for this demo. Classifier
placement experiments are deferred.

A short run and one seed cannot establish the best attainable performance of each
initialization. Shared learning rates may favor one model. Class imbalance,
filename-fallback grouping and PlantVillage's controlled backgrounds also limit
my conclusions. I reserve the test set for final evaluation after fixing the
experimental protocol.

I cite Mohanty, Hughes and Salathé (2016),
[Using Deep Learning for Image-Based Plant Disease Detection](https://doi.org/10.3389/fpls.2016.01419).
The Hugging Face dataset card declares CC BY-SA 3.0; I retain attribution when
sharing dataset images.
