import os
# Aggressively clear any persistent MLflow run state from the environment
for key in list(os.environ.keys()):
    if key.startswith("MLFLOW_"):
        del os.environ[key]

from src.pipeline.training_pipeline import TrainPipeline

pipeline = TrainPipeline()
pipeline.run_pipeline()

# import boto3
# sts = boto3.client("sts")
# print(sts.get_caller_identity())