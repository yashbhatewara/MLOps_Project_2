from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from uvicorn import run as app_run
from fastapi.responses import RedirectResponse

from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipeline.prediction_pipeline import PropertyData, PropertyPredictor
from src.pipeline.training_pipeline import TrainPipeline

import mlflow
import dagshub
dagshub.init(repo_owner='yashbhatewara', repo_name='MLOps_Project_2', mlflow=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="template")


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Area_in_sqft: Optional[float] = None
        self.Beds: Optional[int] = None
        self.Baths: Optional[int] = None
        self.Sqft_per_bed: Optional[float] = None
        self.Total_Rooms: Optional[int] = None
        self.is_high_rise: Optional[int] = None
        self.Location: Optional[str] = None
        self.Type: Optional[str] = None
        self.Furnishing: Optional[str] = None

    async def get_property_data(self):
        form = await self.request.form()
        self.Area_in_sqft = float(form.get("Area_in_sqft"))
        self.Beds = int(form.get("Beds"))
        self.Baths = int(form.get("Baths"))
        self.Sqft_per_bed = float(form.get("Sqft_per_bed", 0))
        self.Total_Rooms = int(form.get("Total_Rooms", 0))
        self.is_high_rise = int(form.get("is_high_rise", 0))
        self.Location = str(form.get("Location", "")).strip()
        self.Type = str(form.get("Type", "")).strip()
        self.Furnishing = str(form.get("Furnishing", "")).strip()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, prediction: float | None = None):
    return templates.TemplateResponse(
        "rent_form.html",
        {"request": request, "prediction": prediction}
    )


@app.post("/")
async def predict(request: Request):
    try:
        form = DataForm(request)
        await form.get_property_data()

        # Re-compute derived features if not provided (though JS adds them)
        sqft_per_bed = form.Area_in_sqft / max(form.Beds, 1)
        total_rooms = form.Beds + form.Baths

        property_data = PropertyData(
            Area_in_sqft=form.Area_in_sqft,
            Beds=form.Beds,
            Baths=form.Baths,
            Sqft_per_bed=sqft_per_bed,
            Total_Rooms=total_rooms,
            is_high_rise=form.is_high_rise,
            Location=form.Location,
            Type=form.Type,
            Furnishing=form.Furnishing
        )

        df = property_data.get_property_input_dataframe()

        predictor = PropertyPredictor()
        prediction = predictor.predict(df, location=form.Location)
        
        import math
        prediction = math.ceil(prediction)

        # store prediction in query param
        return RedirectResponse(
            url=f"/?prediction={prediction}",
            status_code=303
        )

    except Exception as e:
        import traceback
        return {"error": f"Error occurred in python script: [{e}]", "traceback": traceback.format_exc()}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
