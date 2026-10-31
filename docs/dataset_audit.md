# Dataset audit

The audit covered the Hugging Face dataset card, loading script, color manifests
and leaf map before split implementation. Metadata checks included counts,
duplicate paths, class labels and overlapping leaf identifiers. Test images were
not inspected.

Dataset: [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage).
Revision: `9e97599868962bd0079b8db4b7f1efa9185fa1e7`.

The card describes image, image_path, label, crop, disease and leaf_id fields.
The legacy loading script makes the published manifests the basis for image
records. These retain image path, label, leaf identifier and original training
index; crop and disease names come from the class label. The sorted class order
matches the 38 classes in the original audit. Neither published split contains
duplicate paths.

| Metadata check | Count |
|---|---:|
| Published training images | 43,596 |
| Published test images | 10,709 |
| Shared leaf identifiers under upstream rules | 227 |
| Training images excluded for overlap | 290 |
| Fixed validation images, seed 42 | 8,646 |
| Eligible training pool after validation | 34,660 |

| Training group fraction | Selected images, seed 42 |
|---|---:|
| 1% | 467 |
| 3% | 1,193 |
| 10% | 3,615 |
| 30% | 10,530 |

Preserving whole leaf groups gives the nominal 10% subset an actual image fraction
of 10.43%. The groups are nested across fractions, and validation stays fixed for
a given seed. Each experiment saves the class mapping and exact selected indices.

The upstream rules use filename fallbacks for 10,518 training and 2,676 test images.
These groups stay intact, with training groups shared with test conservatively
excluded. Unknown physical-leaf identities remain a limitation. Checks on cached
real metadata after the refactor confirmed known leaf-ID disjointness for all four
fractions, with unchanged class ordering and selected indices.

## Visual inspection in Colab

Visual inspection of real images is still pending. The notebook now includes a
training-only gallery after the image download, with one sampled image per class,
original dimensions and labels. The run directory stores gallery figures, the
sampling seed and image paths in `image_review_samples.json`, and editable notes
in `image_review_notes.md`. The notes remain marked pending until the images have
been examined and observations recorded. This sample review can flag image-quality
or label concerns, but cannot establish the correctness of every disease label.
