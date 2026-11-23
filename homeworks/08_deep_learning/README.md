# Homework 8: Deep Learning

**The notebook to submit is [`08_deep_learning.ipynb`](08_deep_learning.ipynb).** It trains a small PyTorch CNN that classifies curly and straight hair, and holds the answers to Questions 1–6.

## Files

| File | Purpose |
| --- | --- |
| [`08_deep_learning.ipynb`](08_deep_learning.ipynb) | **Homework submission.** |
| [`requirements-cpu.txt`](requirements-cpu.txt) | Pinned CPU dependencies. |
| [`history_baseline.json`](history_baseline.json) | Metrics of the 10 baseline epochs (Questions 3–4). |
| [`history_augmented.json`](history_augmented.json) | Metrics of the 10 augmented epochs (Questions 5–6). |

## How to run

You need [uv](https://docs.astral.sh/uv/), `wget` and `unzip`.

**1. Create the environment** (Python 3.11, separate from the Python 3.13 one at the repo root):

```bash
cd homeworks/08_deep_learning
uv venv --python 3.11
uv pip install --python .venv -r requirements-cpu.txt ipykernel jupyterlab
```

uv warns that Python 3.11 does not match the root project. This is expected.

**2. Get the dataset** (about 100 MB; skip it if `data/straight-curly-data/` already exists at the repo root):

```bash
wget -O ../../data/straight-curly-data.zip https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
unzip -q -o ../../data/straight-curly-data.zip -d ../../data/straight-curly-data
```

**3. Run the notebook:**

```bash
.venv/bin/jupyter lab 08_deep_learning.ipynb
```

In VS Code, open the notebook and select `homeworks/08_deep_learning/.venv` as the kernel.

Skip the `!wget`, `!unzip` and `!uv pip install` cells (done in steps 1 and 2), then run the other cells **once, from top to bottom, on a fresh kernel**. The augmented phase continues from the baseline model, so re-running a training cell changes the results.

The 20 epochs take about 3 minutes on CPU (Apple laptop) and overwrite both history files.

## Results

| Question | Value |
| --- | ---: |
| Q1: loss and output contract | BCEWithLogitsLoss() |
| Q2: trainable parameters | 20,073,473 |
| Q3: median of baseline train accuracy | 0.81 |
| Q4: standard deviation of baseline train loss | 0.139 |
| Q5: mean of augmented evaluation loss | 0.611 |
| Q6: mean of augmented evaluation accuracy (last 5 epochs) | 0.70 |
