import os

# Project Constants
PIPELINE_NAME: str = "src"
ARTIFACT_DIR: str = "artifact"

# Data Ingestion Constants
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
DATA_INGESTION_COLLECTION_NAME: str = "Proj1-Data"

# Database Constants
DATABASE_NAME: str = "Project1"
MONGODB_URL_KEY: str = "MONGODB_URL"

# Shared Constants
SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")
FILE_NAME: str = "properties.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Model Training Constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_R2_SCORE: float = 0.5
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

# Model Pusher Constants
SAVED_MODEL_DIR_NAME: str = "saved_models"
SAVED_MODEL_FILE_PATH: str = os.path.join(SAVED_MODEL_DIR_NAME, MODEL_FILE_NAME)

# Unused constants but needed for compatibility in config_entity
MODEL_BUCKET_NAME = "none"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME = "report.yaml"
DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = "transformed_object"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

# Model Trainer Detail Params
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 0
XGB_RANDOM_STATE = 42

# Target Column
TARGET_COLUMN: str = "Monthly_Rent"

# MLflow Constants — defaults to local mlruns but can be overridden
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT_NAME = "housing_price_model_v1"

# Web application settings (can be overridden via environment)
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
# convert port to int, default 8000
try:
    APP_PORT = int(os.getenv("PORT", os.getenv("APP_PORT", "8000")))
except ValueError:
    APP_PORT = 8000

