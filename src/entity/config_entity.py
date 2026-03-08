import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    trained_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    test_file_path: str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME
    
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    TRAIN_FILE_NAME.replace("csv", "npy"))
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TEST_FILE_NAME.replace("csv", "npy"))
    transformed_object_file_path: str = os.path.join(data_transformation_dir,
                                                     DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCSSING_OBJECT_FILE_NAME)
@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    # this value can be overridden by setting an environment variable
    # e.g. in production: export MODEL_TRAINER_EXPECTED_R2_SCORE=0.5
    expected_r2_score: float = float(os.getenv("MODEL_TRAINER_EXPECTED_R2_SCORE", MODEL_TRAINER_EXPECTED_R2_SCORE))
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH

    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI
    # experiment name may also be overridden; notebooks and pipelines can
    # point to different experiments by setting the env var before running
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME)
    
@dataclass
class ModelPusherConfig:
    model_pusher_dir: str = os.path.join(training_pipeline_config.artifact_dir, "model_pusher")
    saved_model_path: str = SAVED_MODEL_FILE_PATH

@dataclass
class VehiclePredictorConfig:
    model_file_path: str = SAVED_MODEL_FILE_PATH
    model_bucket_name: str = MODEL_BUCKET_NAME

@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        "model_evaluation"
    )
    change_threshold: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE

@dataclass
class ModelPusherConfig:
    model_pusher_dir: str = os.path.join(training_pipeline_config.artifact_dir, "model_pusher")
    saved_model_path: str = SAVED_MODEL_FILE_PATH
