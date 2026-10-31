# Verification record

Verified locally on 2026-10-31 using Python 3.12.13, macOS 26.5.2 ARM64,
PyTorch 2.6.0 and torchvision 0.21.0. CUDA was unavailable. Direct dependencies
are pinned in requirements.txt; the installed environment is captured in
requirements-lock.txt.

## Automated verification

Command:

```bash
MPLCONFIGDIR=/tmp/plant-mpl .venv/bin/python -m pytest -q
```

Result: **15 passed** (7.15 seconds on the final run). Matplotlib emitted 14
Pyparsing deprecation warnings from installed dependencies; there were no test
failures. These warnings do not indicate failed plots.

Coverage includes:

- Fixed, deterministic, nested class-aware training subsets with leaf-group
  disjointness, held-out overlap quarantine and invalid fraction handling.
- RGB conversion, image tensor dimensions and ImageNet normalization.
- ResNet-18 output dimensions, trainable layers and requested pretrained weights
  (the offline unit test mocks the download).
- Synthetic CPU training with parameter updates and best-checkpoint persistence;
  validation does not mutate parameters.
- Known-answer accuracy, macro F1, missing-class treatment and confusion matrices.
- Inference return format, class mapping, crop/disease labels, softmax scores,
  PIL-image and file-path inputs.
- A synthetic ZIP → Hugging Face image dataset → real ResNet-18 CPU training →
  checkpoint reload → validation artifacts → inference integration check. It
  checks that held-out archive images are not extracted.
- Frozen-backbone weights and batch-normalization statistics.
- Comparison-table rejection of unequal optimization settings.
- The full matched runner with synthetic data and offline model initialization:
  both initialization branches are requested, budget/sample counts match, and
  saved split manifests agree. This test does not use real pretrained weights.
- Notebook schema validation, code-cell compilation and absence of fabricated
  execution outputs.

Additional checks passed:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q src tests
MPLCONFIGDIR=/tmp/plant-mpl .venv/bin/python -m src.train --help
MPLCONFIGDIR=/tmp/plant-mpl .venv/bin/python -m src.experiment --help
git diff --check
```

## Live source checks actually performed

1. Read the Hugging Face dataset card, repository tree, published split manifests,
   leaf map and upstream loading script before finalizing the split strategy.
2. Downloaded the pinned color metadata (not the image archive), ran the implemented
   downloader, and reproduced the committed audit and split identity:
   `e38cf4b5ac3f82300983923f373b475a84d6cf0e5d4dea18848f5b763ceea633`.
   All four planned subset fractions were checked for leaf-ID disjointness from
   validation/test. The audit is in dataset_audit.json.
3. Downloaded the official `resnet18-f37072fd.pth` ImageNet weights through
   torchvision into an ignored cache, built the pretrained 38-class model and
   executed a CPU forward pass on synthetic inputs; output shape was `(2, 38)`.
   This was real weight loading, separate from the mocked unit test.

## Not executed

- No Google Colab session or CUDA/GPU training.
- No full PlantVillage image archive download or real-image training/evaluation.
  Archive extraction was exercised with synthetic archives; the real archive
  path/layout and full download integration still need the Colab run.
- No real PlantVillage classification accuracy, F1, prediction examples or
  experimental comparison results. Synthetic fixtures are software checks only.
- No held-out test image decoding or test evaluation.
- No GitHub push; local commits await owner approval before publication.

## Demo completion steps

After the approved GitHub push, open notebooks/progress_demo.ipynb in Colab, select
a GPU runtime, run all cells, and retain the full output directory in Google Drive.
Alternatively upload/extract the source ZIP before publication. Copy observed
metrics and figures into progress_report.md, record the saved environment, and
explain the results and limitations. Never substitute synthetic test values for
research measurements.
