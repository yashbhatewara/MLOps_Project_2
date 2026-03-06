import json
import sys
import os
import pandas as pd

from pandas import DataFrame
from typing import Tuple

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        """
        Initialize DataValidation class.

        :param data_ingestion_artifact: Output of DataIngestion stage
        :param data_validation_config: Configuration for DataValidation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    # ---------------------------------------------------------------------
    # Basic File Reader
    # ---------------------------------------------------------------------
    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys) from e

    # ---------------------------------------------------------------------
    # Column Validation
    # ---------------------------------------------------------------------
    def validate_required_columns(self, df: DataFrame) -> Tuple[bool, str]:
        try:
            required_columns = list(self._schema_config["columns"].keys())
            missing_columns = [
                col for col in required_columns if col not in df.columns
            ]

            if missing_columns:
                message = f"Missing required columns: {missing_columns}"
                logging.info(message)
                return False, message

            return True, ""

        except Exception as e:
            raise MyException(e, sys) from e

    def validate_numerical_categorical_columns(self, df: DataFrame) -> Tuple[bool, str]:
        try:
            missing_numerical = [
                col for col in self._schema_config["numerical_columns"]
                if col not in df.columns
            ]

            missing_categorical = [
                col for col in self._schema_config["categorical_columns"]
                if col not in df.columns
            ]

            errors = []
            if missing_numerical:
                errors.append(f"Missing numerical columns: {missing_numerical}")
            if missing_categorical:
                errors.append(f"Missing categorical columns: {missing_categorical}")

            if errors:
                message = " | ".join(errors)
                logging.info(message)
                return False, message

            return True, ""

        except Exception as e:
            raise MyException(e, sys) from e

    # ---------------------------------------------------------------------
    # Target Validation (Regression Specific)
    # ---------------------------------------------------------------------
    def validate_target_column(self, df: DataFrame) -> Tuple[bool, str]:
        try:
            from src.constants import TARGET_COLUMN
            target_column = TARGET_COLUMN

            if target_column not in df.columns:
                message = f"Target column '{target_column}' missing."
                logging.info(message)
                return False, message

            if df[target_column].isnull().sum() > 0:
                message = f"Target column '{target_column}' contains null values."
                logging.info(message)
                return False, message

            return True, ""

        except Exception as e:
            raise MyException(e, sys) from e

    # ---------------------------------------------------------------------
    # Main Validation Pipeline
    # ---------------------------------------------------------------------
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation process")

            train_df = self.read_data(
                file_path=self.data_ingestion_artifact.trained_file_path
            )
            test_df = self.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )

            validation_error_messages = []

            # ----------------- Train Validation -----------------
            for df, df_name in [(train_df, "Train"), (test_df, "Test")]:

                status, message = self.validate_required_columns(df)
                if not status:
                    validation_error_messages.append(f"{df_name}: {message}")

                status, message = self.validate_numerical_categorical_columns(df)
                if not status:
                    validation_error_messages.append(f"{df_name}: {message}")

                status, message = self.validate_target_column(df)
                if not status:
                    validation_error_messages.append(f"{df_name}: {message}")

            validation_status = len(validation_error_messages) == 0
            final_message = " | ".join(validation_error_messages)

            # -----------------------------------------------------------------
            # Create Validation Artifact
            # -----------------------------------------------------------------
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=final_message,
                validation_report_file_path=self.data_validation_config.validation_report_file_path,
            )

            # Ensure report directory exists
            os.makedirs(
                os.path.dirname(
                    self.data_validation_config.validation_report_file_path
                ),
                exist_ok=True,
            )

            # Write validation report
            validation_report = {
                "validation_status": validation_status,
                "errors": validation_error_messages,
            }

            with open(
                self.data_validation_config.validation_report_file_path,
                "w",
            ) as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation completed successfully")
            logging.info(f"Validation Status: {validation_status}")

            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys) from e