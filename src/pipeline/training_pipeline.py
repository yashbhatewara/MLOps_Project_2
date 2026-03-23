import sys
from src.exception import MyException
from src.logger import logging
import mlflow

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.entity.config_entity import (DataIngestionConfig,
                                          DataValidationConfig,
                                          DataTransformationConfig,
                                          ModelTrainerConfig,
                                          ModelEvaluationConfig,
                                          ModelPusherConfig)
                                          
from src.entity.artifact_entity import (DataIngestionArtifact,
                                            DataValidationArtifact,
                                            DataTransformationArtifact,
                                            ModelTrainerArtifact,
                                            ModelEvaluationArtifact,
                                            ModelPusherArtifact)



class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()


    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Reading data from MongoDB")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Train and test sets created successfully")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component
        """
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config
                                             )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainPipeline class")

            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        try:
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_transformation_config=self.data_transformation_config,
                                                     data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model training
        """
        try:
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config
                                         )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)
    
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting modle evaluation
        """
        try:
            model_evaluation = ModelEvaluation(model_eval_config=self.model_evaluation_config,
                                               data_ingestion_artifact=data_ingestion_artifact,
                                               model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys)
        
    def start_model_pusher(self, model_trainer_artifact: ModelTrainerArtifact) -> ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for starting model pushing
        """
        try:
            model_pusher = ModelPusher(model_trainer_artifact=model_trainer_artifact,
                                       model_pusher_config=self.model_pusher_config
                                       )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        and logging results to both local and (if provided) DagsHub remote.
        """
        try:
            from src.constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
            import os
            import dagshub
            
            # --- PHASE 1: PREPARATION ---
            # Clear any existing MLflow environment state
            for key in ["MLFLOW_RUN_ID", "MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID"]:
                if key in os.environ:
                    del os.environ[key]

            # Determine logging targets
            tracking_targets = []
            
            # 1. Local logging is always enabled
            local_uri = MLFLOW_TRACKING_URI
            if not local_uri.startswith(("http", "file")):
                local_uri = f"file:///{os.path.abspath(local_uri)}"
            tracking_targets.append(("Local", local_uri))

            # 2. DagsHub logging if token exists
            dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
            if dagshub_token:
                os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
                os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
                remote_uri = "https://dagshub.com/yashbhatewara/MLOps_Project_2.mlflow"
                tracking_targets.append(("DagsHub", remote_uri))
                
                try:
                    dagshub.init(repo_owner='yashbhatewara', repo_name='MLOps_Project_2', mlflow=False)
                except:
                    pass

            # --- PHASE 2: CORE EXECUTION (Run ONCE) ---
            # Ingestion -> Validation -> Transformation -> Training
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact, 
                data_validation_artifact=data_validation_artifact
            )
            
            # This will log to whatever is currently set as tracking URI (we set Local first)
            mlflow.set_tracking_uri(tracking_targets[0][1])
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            # Model rejection check
            if not model_trainer_artifact.is_model_accepted:
                logging.warning("Model did not meet acceptance threshold; continuing to log metrics anyway")
            else:
                model_pusher_artifact = self.start_model_pusher(model_trainer_artifact=model_trainer_artifact)
                logging.info(f"Model pushed locally. Artifact: {model_pusher_artifact}")

            # --- PHASE 3: DUAL LOGGING ---
            # Now iterate through targets and ensure everything is logged
            from src.constants import (XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_MAX_DEPTH, 
                                      XGB_RANDOM_STATE)
            from src.utils.main_utils import load_object

            for name, uri in tracking_targets:
                logging.info(f"Logging results to {name} MLflow at {uri}...")
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
                
                with mlflow.start_run(run_name=f"Pipeline_{name}"):
                    # 1. Component Status Params
                    mlflow.log_param("data_ingestion", "Completed")
                    mlflow.log_param("data_validation", data_validation_artifact.validation_status)
                    mlflow.log_param("data_transformation", "Completed")
                    mlflow.log_param("model_trainer", "Completed")
                    
                    # 2. Hyperparameters
                    mlflow.log_params({
                        "n_estimators": XGB_N_ESTIMATORS,
                        "learning_rate": XGB_LEARNING_RATE,
                        "max_depth": XGB_MAX_DEPTH,
                        "random_state": XGB_RANDOM_STATE
                    })

                    # 3. Metrics
                    metrics = model_trainer_artifact.metric_artifact
                    mlflow.log_metrics({
                        "r2_score": metrics.r2_score,
                        "rmse": metrics.rmse,
                        "mae": metrics.mae,
                        "model_accepted": 1 if model_trainer_artifact.is_model_accepted else 0
                    })

                    # 4. Model Registry & Artifacts
                    # Load the model package to log it formally
                    model_pkg = load_object(model_trainer_artifact.trained_model_file_path)
                    model_obj = model_pkg["trained_model"]
                    
                    mlflow.sklearn.log_model(
                        model_obj, 
                        "model", 
                        registered_model_name="housing_price_model"
                    )
                    mlflow.log_artifact(model_trainer_artifact.trained_model_file_path, artifact_path="model_package")
                    
                    # 5. Visualizations
                    import os
                    viz_dir = getattr(model_trainer_artifact, "visualizations_dir", None)
                    if viz_dir and os.path.exists(viz_dir):
                        logging.info(f"Syncing plots from {viz_dir} to {name}")
                        mlflow.log_artifacts(viz_dir, artifact_path="plots")
                    
                    logging.info(f"Successfully synced results to {name}")

        except Exception as e:
            raise MyException(e, sys)

