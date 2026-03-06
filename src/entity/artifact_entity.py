from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    validation_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str

@dataclass
class RegressionMetricArtifact:
    r2_score: float
    rmse: float
    mae: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: RegressionMetricArtifact
    is_model_accepted: bool

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    previous_model_r2: float
    new_model_r2: float
    improvement: float