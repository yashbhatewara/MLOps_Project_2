import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Read CSV
    # ------------------------------------------------------------------
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Create Transformer Object
    # ------------------------------------------------------------------
    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Creating data transformer object")

            num_features = self._schema_config["num_features"]
            mm_columns = self._schema_config["mm_columns"]

            numeric_transformer = StandardScaler()
            minmax_transformer = MinMaxScaler()

            preprocessor = ColumnTransformer(
                transformers=[
                    ("standard_scaler", numeric_transformer, num_features),
                    ("minmax_scaler", minmax_transformer, mm_columns),
                ],
                remainder="passthrough"
            )

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor)
                ]
            )

            logging.info("Data transformer object created successfully")
            return pipeline

        except Exception as e:
            raise MyException(e, sys)

    # ------------------------------------------------------------------
    # Main Transformation Logic
    # ------------------------------------------------------------------
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation started")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train & test
            train_df = self.read_data(
                file_path=self.data_ingestion_artifact.trained_file_path
            )
            test_df = self.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )

            logging.info("Train and Test data loaded successfully")

            # Filter features based on schema
            feature_cols = self._schema_config["num_features"] + self._schema_config["mm_columns"] + self._schema_config["passthrough_columns"]

            target_feature_train_df = train_df[TARGET_COLUMN].astype(float)
            target_feature_test_df = test_df[TARGET_COLUMN].astype(float)

            # --- Target-encode Location -------------------------------------------
            # Compute mean rent per location from training data only
            location_encoding_map: dict = (
                train_df.groupby("Location")[TARGET_COLUMN]
                .mean()
                .to_dict()
            )
            global_mean: float = float(target_feature_train_df.mean())

            train_location_encoded = (
                train_df["Location"]
                .map(location_encoding_map)
                .fillna(global_mean)
                .values
                .reshape(-1, 1)
            )
            test_location_encoded = (
                test_df["Location"]
                .map(location_encoding_map)
                .fillna(global_mean)
                .values
                .reshape(-1, 1)
            )
            logging.info(f"Location target-encoded. Unique locations: {len(location_encoding_map)}")
            # -----------------------------------------------------------------------

            input_feature_train_df = train_df[feature_cols]
            input_feature_test_df = test_df[feature_cols]

            logging.info("Separated input and target columns")

            # Get preprocessing pipeline
            preprocessor = self.get_data_transformer_object()

            # Fit on train, transform both
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Append target-encoded Location column to feature arrays
            input_feature_train_arr = np.hstack([input_feature_train_arr, train_location_encoded])
            input_feature_test_arr = np.hstack([input_feature_test_arr, test_location_encoded])

            # Attach encoding artifacts to preprocessor so model_trainer can persist them
            preprocessor.location_encoding_map = location_encoding_map
            preprocessor.location_global_mean = global_mean

            logging.info("Feature transformation completed")

            # Convert targets to numpy
            target_feature_train_arr = target_feature_train_df.values
            target_feature_test_arr = target_feature_test_df.values

            # Concatenate features + target
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info("Concatenated features and target")

            # Save transformation object
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            # Save transformed arrays
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            logging.info("Transformation artifacts saved successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

        except Exception as e:
            raise MyException(e, sys)