import sys
import os
import numpy as np
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object
from src.constants import SAVED_MODEL_FILE_PATH


class PropertyData:
    def __init__(
        self,
        Area_in_sqft,
        Beds,
        Baths,
        Sqft_per_bed,
        Total_Rooms,
        is_high_rise,
        Location: str = "",
        Type: str = "",
        Furnishing: str = ""
    ):
        try:
            self.Area_in_sqft = Area_in_sqft
            self.Beds = Beds
            self.Baths = Baths
            self.Sqft_per_bed = Sqft_per_bed
            self.Total_Rooms = Total_Rooms
            self.is_high_rise = is_high_rise
            self.Location = Location
            self.Type = Type
            self.Furnishing = Furnishing
        except Exception as e:
            raise MyException(e, sys)

    def get_property_data_as_dict(self):
        # Return only features used by current model
        return {
            "Area_in_sqft": [self.Area_in_sqft],
            "Beds": [self.Beds],
            "Baths": [self.Baths],
            "Sqft_per_bed": [self.Sqft_per_bed],
            "Total_Rooms": [self.Total_Rooms],
            "is_high_rise": [self.is_high_rise],
        }

    def get_property_input_dataframe(self) -> DataFrame:
        return DataFrame(self.get_property_data_as_dict())


class PropertyPredictor:
    def __init__(self):
        try:
            self.model_package = load_object(SAVED_MODEL_FILE_PATH)
            self.preprocessing_object = self.model_package["preprocessing_object"]
            self.model = self.model_package["trained_model"]
            self.location_encoding_map: dict = self.model_package.get("location_encoding_map", {})
            self.location_global_mean: float = self.model_package.get("location_global_mean", 0.0)
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: DataFrame, location: str):
        try:
            # Apply numeric preprocessing
            transformed = self.preprocessing_object.transform(dataframe)

            # Append target-encoded Location as the last feature column
            encoded_location = self.location_encoding_map.get(
                location.strip(), self.location_global_mean
            )
            location_arr = np.array([[encoded_location]])
            transformed = np.hstack([transformed, location_arr])

            pred_log = self.model.predict(transformed)
            prediction = float(np.expm1(pred_log)[0])
            return prediction
        except Exception as e:
            raise MyException(e, sys)
