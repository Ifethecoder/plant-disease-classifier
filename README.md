# Plant disease classification

Minerva University Deep Learning progress demo investigating: **How does ImageNet
pretraining affect plant disease classification performance when limited labeled
training data is available?**

Implementation in progress. Compare randomly initialized and ImageNet-pretrained
ResNet-18 on the same leaf-grouped training subset and validation set. The published
test split is reserved for final evaluation. No GPU results have been produced yet.

Use Python 3.11 or 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```
