# Homework 5: Deployment

**The notebook to submit is [`05_deployment.ipynb`](05_deployment.ipynb).** It holds the answers to Questions 1–6.

Every other file in this folder supports the homework. The notebook uses them to train, load, serve and containerize the lead-scoring model.

## Files

| File | Purpose |
| --- | --- |
| [`05_deployment.ipynb`](05_deployment.ipynb) | **Homework submission.** Questions 1–6 with commands and answers. |
| [`train.py`](train.py) | Trains `DictVectorizer` + `LogisticRegression` as one sklearn pipeline and saves it to `pipeline.bin`. |
| [`pipeline.bin`](pipeline.bin) | The trained pipeline, saved with pickle (a single object, not a `(dv, model)` tuple). |
| [`model.py`](model.py) | Feature lists and `normalize_record()`, which fills missing values the same way as training. |
| [`feature_defaults.json`](feature_defaults.json) | Training-set medians used for missing numeric features. |
| [`model_metadata.json`](model_metadata.json) | Dataset, model settings, scikit-learn version and SHA-256 checksums. |
| [`predict.py`](predict.py) | FastAPI service with `GET /health` and `POST /predict`. |
| [`q3_script.py`](q3_script.py) | Question 3: loads `pipeline.bin` and scores one lead. |
| [`q6_test.py`](q6_test.py) | Questions 4 and 6: sends a lead to the running API. |
| [`smoke_test.py`](smoke_test.py) | Quick check that the model loads and gives the same result twice. |
| [`Dockerfile`](Dockerfile) | Container image (Python `3.11.15-slim-bookworm`, uv `0.10.11`). |
| `pyproject.toml`, `uv.lock`, `.python-version` | Locked environment for `uv`. |

## How to run

```bash
# Local environment
uv sync
uv run python smoke_test.py
uv run python q3_script.py

# Run the API (Question 4)
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
python q6_test.py            # in another terminal

# Run the container (Questions 5–6)
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
python q6_test.py
curl -s http://localhost:9696/health
```
