import sys
from typing import Tuple
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact
)
from src.constants import *


class ModelTrainer:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    # ------------------------------------------------------------------
    # Train Model & Generate Report
    # ------------------------------------------------------------------
    def get_model_object_and_report(
        self,
        train: np.array,
        test: np.array
    ) -> Tuple[object, RegressionMetricArtifact]:

        try:
            logging.info("Splitting input and target columns")

            X_train, y_train = train[:, :-1].astype(float), train[:, -1].astype(float)
            X_test, y_test = test[:, :-1].astype(float), test[:, -1].astype(float)

            # Optional: Log transform target (if used in notebook)
            logging.info("Applying log1p transformation to target")
            y_train_log = np.log1p(y_train)
            y_test_log = np.log1p(y_test)

            logging.info("Initializing XGBRegressor")

            model = XGBRegressor(
                n_estimators=XGB_N_ESTIMATORS,
                learning_rate=XGB_LEARNING_RATE,
                max_depth=XGB_MAX_DEPTH,
                random_state=XGB_RANDOM_STATE,
                n_jobs=-1
            )

            logging.info("Training model...")
            model.fit(X_train, y_train_log)
            logging.info("Model training completed")

            logging.info("Generating predictions")

            y_train_pred_log = model.predict(X_train)
            y_test_pred_log = model.predict(X_test)

            # Reverse log transform
            y_train_pred = np.expm1(y_train_pred_log)
            y_test_pred = np.expm1(y_test_pred_log)

            # Evaluation metrics
            r2 = r2_score(y_test, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae = mean_absolute_error(y_test, y_test_pred)

            logging.info(f"R2 Score: {r2}")
            logging.info(f"RMSE: {rmse}")
            logging.info(f"MAE: {mae}")

            metric_artifact = RegressionMetricArtifact(
                r2_score=r2,
                rmse=rmse,
                mae=mae
            )

            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    # ------------------------------------------------------------------
    # Main Trainer Pipeline
    # ------------------------------------------------------------------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        logging.info("Entered initiate_model_trainer method")

        try:
            print("------------------------------------------------------")
            print("Starting Model Trainer Component")

            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            logging.info("Train & Test arrays loaded")

            model, metric_artifact = self.get_model_object_and_report(
                train=train_arr,
                test=test_arr
            )

            logging.info("Model training and evaluation completed")

            # Model acceptance logic – compute a flag but do not raise
            is_accepted = metric_artifact.r2_score >= self.model_trainer_config.expected_r2_score
            if not is_accepted:
                logging.warning(
                    f"Model R2 score {metric_artifact.r2_score} is less than expected "
                    f"{self.model_trainer_config.expected_r2_score}. "
                    "Run will continue and metrics will be logged."
                )
            else:
                logging.info("Model accepted based on R2 threshold")

            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            # Extract location encoding artifacts saved by DataTransformation
            location_encoding_map = getattr(preprocessing_obj, "location_encoding_map", {})
            location_global_mean = getattr(preprocessing_obj, "location_global_mean", 0.0)
            logging.info(f"Location encoding map loaded: {len(location_encoding_map)} entries")

            # Save combined model package
            model_package = {
                "preprocessing_object": preprocessing_obj,
                "trained_model": model,
                "location_encoding_map": location_encoding_map,
                "location_global_mean": location_global_mean,
            }

            save_object(
                self.model_trainer_config.trained_model_file_path,
                model_package
            )

            logging.info("Final model package saved successfully")

            # ------------------------------------------------------------------
            # MLflow Logging
            # ------------------------------------------------------------------
            try:
                logging.info("Logging to MLflow...")
                uri = self.model_trainer_config.mlflow_tracking_uri
                logging.info(f"Using MLflow tracking URI: {uri if uri else 'default (mlruns)'}")
                if uri:
                    # Using file: prefix makes MLflow more reliable with local folders
                    tracking_uri = f"file:///{os.path.abspath(uri)}"
                    logging.info(f"Setting MLflow tracking URI to: {tracking_uri}")
                    mlflow.set_tracking_uri(tracking_uri)
                    mlflow.set_registry_uri(tracking_uri)

                # Explicitly set the experiment name to avoid defaulting to experiment ID 0
                exp_name = self.model_trainer_config.mlflow_experiment_name
                logging.info(f"Setting MLflow experiment: {exp_name}")
                mlflow.set_experiment(exp_name)

                with mlflow.start_run():
                    mlflow.log_param("n_estimators", XGB_N_ESTIMATORS)
                    mlflow.log_param("learning_rate", XGB_LEARNING_RATE)
                    mlflow.log_param("max_depth", XGB_MAX_DEPTH)
                    mlflow.log_param("random_state", XGB_RANDOM_STATE)

                    mlflow.log_metric("r2_score", metric_artifact.r2_score)
                    mlflow.log_metric("rmse", metric_artifact.rmse)
                    mlflow.log_metric("mae", metric_artifact.mae)

                    mlflow.sklearn.log_model(model, "model")

                logging.info("MLflow logging completed")
            except Exception as mlflow_err:
                logging.warning(f"MLflow logging skipped due to error: {mlflow_err}")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
                is_model_accepted=is_accepted
            )

            logging.info(f"ModelTrainerArtifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e