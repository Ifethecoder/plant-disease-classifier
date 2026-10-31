# Experiment outputs

Settings, split indices, checkpoints, losses, validation metrics, comparison tables
and figures are saved here for local runs, or in Google Drive when running Colab.
Generated files in this folder and model checkpoints are ignored by Git.

The [executed progress notebook](../notebooks/progress_demo.ipynb) includes the
saved Tesla T4 experiment: random initialization achieved 58.69% validation
accuracy and 0.4853 macro F1; ImageNet pretraining achieved 82.73% and 0.7926.
Both selected epoch 5. These are rounded, single-seed development results from
3,615 training and 8,646 validation images. The test set remains reserved.

The notebook retains logs and figures. Separate checkpoints and result files are
saved to its configured Drive run directory. Synthetic CPU checks remain software
verification, separate from this PlantVillage experiment.
