import os
import mlflow
import dagshub
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_mlflow")

token = "9f3ef659d4ca3e1e95191e17c4e0d925a53b907f"
os.environ["MLFLOW_TRACKING_USERNAME"] = token
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# Force fresh state
for key in ["MLFLOW_RUN_ID", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID"]:
    if key in os.environ:
        del os.environ[key]

uri = "https://dagshub.com/yashbhatewara/MLOps_Project_2.mlflow"
mlflow.set_tracking_uri(uri)
logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

exp_name = "debug_experiment_test"
mlflow.set_experiment(exp_name)
logger.info(f"Experiment: {exp_name}")

try:
    logger.info("Starting run...")
    with mlflow.start_run() as run:
        logger.info(f"Run started successfully. ID: {run.info.run_id}")
        mlflow.log_param("test_param", "hello")
        logger.info("Logged param successfully")
except Exception as e:
    logger.error(f"Failed to start/log run: {e}")
    import traceback
    logger.error(traceback.format_exc())
