import sys
import os
import shutil
from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            trained_model_path = self.model_trainer_artifact.trained_model_file_path
            
            # Create the path for saving model
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            
            # Copy the trained model from artifact directory to saved_models directory
            shutil.copy(src=trained_model_path, dst=saved_model_path)
            
            # Also copy to model pusher artifact directory for tracking
            model_pusher_dir = self.model_pusher_config.model_pusher_dir
            os.makedirs(model_pusher_dir, exist_ok=True)
            artifact_model_path = os.path.join(model_pusher_dir, os.path.basename(saved_model_path))
            shutil.copy(src=trained_model_path, dst=artifact_model_path)

            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=saved_model_path,
                model_pusher_dir=model_pusher_dir
            )
            
            logging.info(f"Model pushed to {saved_model_path}")
            return model_pusher_artifact
            
        except Exception as e:
            raise MyException(e, sys)
