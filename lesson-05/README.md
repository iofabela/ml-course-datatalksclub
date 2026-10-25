# Lesson 05 - Churn Prediction Model Deployment

This lesson takes the customer churn model built in lesson 03 out of a notebook and turns it into a web service. Other systems can call it over HTTP to get predictions.

## Why this project exists

A model that only runs inside a notebook can't be used by the rest of a company. Say a marketing team wants to send promo emails to customers who are likely to leave. Their system has to ask the model for a churn probability automatically, for thousands of customers, without anyone opening a notebook.

To make that possible, this lesson:

1. **Saves the model** with `pickle`, so it isn't retrained for every prediction.
2. **Serves it with Flask** on a `/predict` endpoint.
3. **Runs it on a production server** (`gunicorn`) instead of the Flask development server.
4. **Locks the dependencies** with `uv` (`pyproject.toml`), so the service runs with the versions it was tested with.
5. **Packages it with Docker**, so it runs the same way on any machine.
6. **(Optional) Deploys it to the cloud** with AWS Elastic Beanstalk.

Full lesson notes: [05-deployment.md](05-deployment.md).

## Project structure

| File | Purpose |
|------|---------|
| [notebook-train.ipynb](notebook-train.ipynb) | Notebook where the training code was developed |
| [train.py](train.py) | Trains the model, validates it with K-Fold and saves it to `model_C=1.0.bin` |
| `model_C=1.0.bin` | Pickled `(DictVectorizer, LogisticRegression)` tuple |
| [predict.py](predict.py) | Flask web service with the `/predict` endpoint |
| [predict_test.py](predict_test.py) / [predict_test.ipynb](predict_test.ipynb) | Client that sends a customer to the service and prints the result |
| [ping.py](ping.py) | Minimal Flask example (`/ping` returns `PONG`) |
| [Dockerfile](Dockerfile) / [.dockerignore](.dockerignore) | Container image for the service |
| [pyproject.toml](pyproject.toml) | Project dependencies (`flask`, `gunicorn`, `scikit-learn`) |
| [images/](images/) | Diagrams used in the lesson notes |

## Requirements

- Python `>= 3.15`
- [uv](https://docs.astral.sh/uv/)
- Docker (optional, for the container)
- `requests` (only for the test client)

Install the dependencies from the `lesson-05/` folder:

```bash
uv sync
```

## How to use it

### 1. Train the model (optional)

`model_C=1.0.bin` is already in the repo. To regenerate it:

```bash
uv run python train.py
```

`train.py` reads the dataset from `./data/data-week-3.csv`, relative to the folder you run it from. You can change these parameters at the top of the script:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `C` | `1.0` | Inverse regularization strength of the logistic regression |
| `n_splits` | `5` | Number of K-Fold splits used for validation |
| `output_file` | `model_C={C}.bin` | File name the model is saved to |

### 2. Run the web service

Development server (Flask):

```bash
uv run python predict.py
```

Production server (gunicorn):

```bash
uv run gunicorn --bind=0.0.0.0:9696 predict:app
```

On Windows, use `waitress` instead: `waitress-serve --listen=0.0.0.0:9696 predict:app`.

The service listens on **port 9696**.

### 3. Run with Docker

```bash
docker build -t churn-prediction .
docker run -it --rm -p 9696:9696 churn-prediction:latest
```

- `-t churn-prediction`: names the image.
- `-it`: keeps the terminal attached (stop with Ctrl-C).
- `--rm`: removes the container when it stops.
- `-p 9696:9696`: maps the host port to the container port (`host:container`).

### 4. Test the service

With the service running (locally or in Docker), run this in another terminal:

```bash
uv run python predict_test.py
```

Expected output (the values depend on the customer data):

```
{'churn': False, 'churn_probability': 0.32...}
not sending promo email to xyz-123
```

To check that Flask works at all, run `python ping.py` and then `curl http://localhost:9696/ping`.

## API reference

### `POST /predict`

**Request body (JSON)**: one customer record with the same fields the model was trained on.

| Field | Type | Example values |
|-------|------|----------------|
| `gender` | string | `female`, `male` |
| `seniorcitizen` | int | `0`, `1` |
| `partner` | string | `yes`, `no` |
| `dependents` | string | `yes`, `no` |
| `phoneservice` | string | `yes`, `no` |
| `multiplelines` | string | `yes`, `no`, `no_phone_service` |
| `internetservice` | string | `dsl`, `fiber_optic`, `no` |
| `onlinesecurity` | string | `yes`, `no`, `no_internet_service` |
| `onlinebackup` | string | `yes`, `no`, `no_internet_service` |
| `deviceprotection` | string | `yes`, `no`, `no_internet_service` |
| `techsupport` | string | `yes`, `no`, `no_internet_service` |
| `streamingtv` | string | `yes`, `no`, `no_internet_service` |
| `streamingmovies` | string | `yes`, `no`, `no_internet_service` |
| `contract` | string | `month-to-month`, `one_year`, `two_year` |
| `paperlessbilling` | string | `yes`, `no` |
| `paymentmethod` | string | `electronic_check`, `mailed_check`, `bank_transfer_(automatic)`, `credit_card_(automatic)` |
| `tenure` | int | months as a customer, e.g. `24` |
| `monthlycharges` | float | e.g. `29.85` |
| `totalcharges` | float | e.g. `716.4` |

String values must be lowercase, with spaces replaced by `_`. This is the same cleaning `train.py` applies.

Example:

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"gender": "female", "seniorcitizen": 0, "partner": "yes", "dependents": "no",
       "phoneservice": "no", "multiplelines": "no_phone_service", "internetservice": "dsl",
       "onlinesecurity": "no", "onlinebackup": "yes", "deviceprotection": "no",
       "techsupport": "no", "streamingtv": "no", "streamingmovies": "no",
       "contract": "month-to-month", "paperlessbilling": "yes",
       "paymentmethod": "electronic_check", "tenure": 1,
       "monthlycharges": 29.85, "totalcharges": 29.85}'
```

**Response (JSON)**:

| Field | Type | Meaning |
|-------|------|---------|
| `churn_probability` | float | Probability that the customer churns (0-1) |
| `churn` | bool | `true` if `churn_probability >= 0.5` |

## Deploying to the cloud (optional)

The Docker image can be deployed to AWS Elastic Beanstalk:

```bash
eb init -p docker -r eu-north-1 churn-serving   # set up the EB application
eb local run --port 9696                        # test it locally first
eb create churn-serving-env                     # create the environment in AWS
eb terminate churn-serving-env                  # delete it when you're done (to stop charges)
```

After `eb create`, put the environment URL in `predict_test.py` (no port needed). The endpoint is public, so add authentication before using it for real. See section 5.7 in [05-deployment.md](05-deployment.md) for details.

## Notes

- **Only unpickle files you trust.** Loading a pickle file can run code. Only load model files you created yourself.
- The model is loaded once, when the service starts, not on every request.
- NumPy values are cast to native `float`/`bool` before they're returned, because Flask can't serialize NumPy types to JSON.
