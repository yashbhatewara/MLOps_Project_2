from src.pipeline.training_pipeline import TrainPipeline

pipeline = TrainPipeline()
pipeline.run_pipeline()

# import boto3
# sts = boto3.client("sts")
# print(sts.get_caller_identity())