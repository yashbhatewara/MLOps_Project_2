import sys

import numpy as np
from sklearn.metrics import r2_score

from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact
)
from src.entity.config_entity import ModelEvaluationConfig
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object, load_numpy_array_data


class ModelEvaluation:
    def __init__(self,
                 model_eval_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):

        self.model_eval_config = model_eval_config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation")

            # If model trainer already rejected model → stop
            if not self.model_trainer_artifact.is_model_accepted:
                logging.info("Model rejected at training stage.")
                return ModelEvaluationArtifact(
                    is_model_accepted=False,
                    previous_model_r2=0.0,
                    new_model_r2=self.model_trainer_artifact.metric_artifact.r2_score,
                    improvement=0.0
                )

            new_model_package = load_object(
                self.model_trainer_artifact.trained_model_file_path
            )
            new_model = new_model_package["trained_model"]

            # Load transformed test data
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # New model performance
            y_pred_log = new_model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            new_r2 = r2_score(y_test, y_pred)

            logging.info(f"New model R2: {new_r2}")

            # Try loading previous production model
            try:
                previous_model_package = load_object("production_model.pkl")
                previous_model = previous_model_package["trained_model"]

                y_prev_log = previous_model.predict(X_test)
                y_prev = np.expm1(y_prev_log)
                prev_r2 = r2_score(y_test, y_prev)

                logging.info(f"Previous model R2: {prev_r2}")

            except:
                logging.info("No previous production model found. Accepting new model.")
                return ModelEvaluationArtifact(
                    is_model_accepted=True,
                    previous_model_r2=0.0,
                    new_model_r2=new_r2,
                    improvement=new_r2
                )

            improvement = new_r2 - prev_r2
            is_accepted = improvement > self.model_eval_config.change_threshold

            logging.info(f"Improvement: {improvement}")
            logging.info(f"Model Accepted: {is_accepted}")

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                previous_model_r2=prev_r2,
                new_model_r2=new_r2,
                improvement=improvement
            )

        except Exception as e:
            raise MyException(e, sys)