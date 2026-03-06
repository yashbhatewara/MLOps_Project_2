# MLOps_Project_2

This repository contains an end-to-end training pipeline that reads data
from a MongoDB Atlas cluster, processes it, and trains a regression model
while tracking experiments with MLflow.

## Getting Started

Before running the pipeline you must configure the MongoDB connection string
in an environment variable named `MONGODB_URL`. Example:

```powershell
$env:MONGODB_URL='mongodb+srv://user:pass@cluster0.mongodb.net/mydb?retryWrites=true&w=majority'
```

Two additional environment variables may assist when debugging TLS issues:

* `MONGODB_TLS` – set to `false` to disable TLS (use only for diagnosis).
* `MONGODB_TLS_ALLOW_INVALID_CERTS` – set to `true` to ignore certificate
  validation errors.

## Testing the database connection

A helper script `check_mongo.py` is provided at the project root. Run:

```powershell
python check_mongo.py
```

It will attempt to establish a connection and print success or the error
returned by the driver. This is helpful when the training pipeline fails early
with SSL handshake errors.

## Running the pipeline

Once connectivity is confirmed, simply run:

```powershell
python demo.py
```

The training components will log progress to `logs/` and MLflow will track
experiments as long as the `mlflow` package is installed.

### MLflow tracking URI

The pipeline defaults to using `http://localhost:5000` as the MLflow
tracking URI (see `src/constants/__init__.py`), which assumes you have an
MLflow server running on that address. If you prefer to use the local file
backend (e.g. for quick testing) you can override the URI via one of the
following methods:

```powershell
# environment variable (recommended)
$env:MLFLOW_TRACKING_URI='sqlite:///C:/path/to/mlflow.db'
# or
$env:MLFLOW_TRACKING_URI='file:///C:/path/to/artifacts'
```

or modify `src/constants/__init__.py` directly.  The code now calls
`mlflow.set_tracking_uri(uri)` before each run, so the URI is always
honored. Remember that if you use a filesystem or sqlite URI, MLflow will not
support an `mlflow-artifacts` URI – you must either use an HTTP server or
ensure your artifact locations are also filesystem paths.

If you see errors like:

```
When an mlflow-artifacts URI was supplied, the tracking URI must be a valid http or https URI,
```

it means your tracking URI is something like `sqlite:///…` while your
experiment/registry is trying to write artifacts to an HTTP location. Either
run a server (`mlflow server --backend-store-uri sqlite:///mlflow.db`) or
use matching file paths for both tracking and artifacts.

By default the pipeline will not “hamper” your model; changing the tracking
URI only affects where metadata and artifacts are logged, not the model
itself.

### Separating notebook and pipeline runs

The pipeline and the notebook share the same default MLflow experiment name
(`housing_price_prediction`). If you train a very good model in the notebook
then run the pipeline later with a lower score, the notebook results appear
"better" because they belong to the same experiment. To keep things
separate you can set the experiment name via an environment variable:

```powershell
# notebook run
$env:MLFLOW_EXPERIMENT_NAME='notebook'
python notebook.ipynb

# production pipeline
$env:MLFLOW_EXPERIMENT_NAME='production'
python demo.py
```

The code now reads `MLFLOW_EXPERIMENT_NAME` when the `ModelTrainerConfig` is
constructed, so whichever value you export is the one used for that session.

Similarly the acceptance threshold (`expected_r2_score`) can be overridden
from the environment by setting `MODEL_TRAINER_EXPECTED_R2_SCORE`. For
development you might keep the default `0.6`, but in production you could
lower it to `0.55` without modifying the source.

These environment variables are applied when the configuration objects are
created, giving you full control over experiment separation and model
acceptance for different contexts.

## Running the web application

The FastAPI server spawned by `app.py` listens on `APP_HOST`/`APP_PORT`, which
default to `0.0.0.0:8000`. You can override these with environment variables
if the port is already in use:

```powershell
$env:APP_PORT=8001
python app.py
```

If you see an error like

```
[Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000):
[winerror 10048] only one usage of each socket address (protocol/network address/port) is normally permitted
```

it means another process is already bound to that port. Either kill the
existing process (e.g. `lsof -i :8000` on Unix, or use Task Manager on
Windows) or choose a different port with `APP_PORT`.
