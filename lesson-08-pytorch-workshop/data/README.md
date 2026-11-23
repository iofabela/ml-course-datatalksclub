# Data

This folder holds the dataset used in [pytorch_notebook.ipynb](../pytorch_notebook.ipynb).
The dataset itself is **not committed** to this repo. Download it yourself.

## Dataset

[clothing-dataset-small](https://github.com/alexeygrigorev/clothing-dataset-small) by Alexey Grigorev, a small version of the clothing dataset with 10 classes split into `train`, `validation` and `test`.

## Download (shallow clone)

You only need the files, not the repository history. Use `--depth 1` to make a **shallow clone** that downloads just the latest commit. It's faster and smaller.

From the `lesson-08-pytorch-workshop/` folder:

```bash
git clone --depth 1 https://github.com/alexeygrigorev/clothing-dataset-small.git ./data/clothing-dataset-small
```

Or from a notebook cell:

```python
!git clone --depth 1 https://github.com/alexeygrigorev/clothing-dataset-small.git ./data/clothing-dataset-small
```

Optional: remove the nested `.git` folder so it isn't treated as a separate repository:

```bash
rm -rf ./data/clothing-dataset-small/.git
```

## Expected structure

```
data/
└── clothing-dataset-small/
    ├── train/
    ├── validation/
    └── test/
```

## Flags reference

| Flag | What it does |
|------|--------------|
| `--depth 1` | Downloads only the latest commit (no history) |
| `--single-branch` | Fetches only the default branch (implied by `--depth`) |
| `--filter=blob:none` | Partial clone: downloads file contents on demand (not needed here) |
