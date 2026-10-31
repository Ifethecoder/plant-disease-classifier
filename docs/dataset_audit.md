# My dataset audit

I inspected the Hugging Face dataset card, loading script, color manifests and leaf
map before implementing my splits. I checked counts, duplicate paths, class labels
and overlapping leaf identifiers using metadata; I did not inspect test images.

Dataset: [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage).
Revision: `9e97599868962bd0079b8db4b7f1efa9185fa1e7`.

The card describes image, image_path, label, crop, disease and leaf_id fields.
The repository uses a legacy script, so I build image records from its manifests.
I retain image path, label, leaf identifier and original training index; crop and
disease names come from the class label. My sorted class order matches the 38
classes in the original audit. Neither published split contains duplicate paths.

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

I preserve whole leaf groups, making the 10% group's actual image fraction 10.43%.
The groups are nested across fractions and validation stays fixed for a given seed.
I save the class mapping and exact selected indices with each experiment.

The upstream rules use filename fallbacks for 10,518 training and 2,676 test images.
I keep these groups intact and conservatively exclude training groups shared with
test, but unknown physical-leaf identities remain a limitation. I checked all four
fractions for known leaf-ID disjointness using the cached real metadata after the
refactor; the class ordering and selected indices were unchanged.
