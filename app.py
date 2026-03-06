from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from uvicorn import run as app_run

from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipeline.prediction_pipeline import PropertyData, PropertyPredictor
from src.pipeline.training_pipeline import TrainPipeline

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

    async def get_property_data(self):
        form = await self.request.form()
        self.Area_in_sqft = float(form.get("Area_in_sqft"))
        self.Beds = int(form.get("Beds"))
        self.Baths = int(form.get("Baths"))
        self.Sqft_per_bed = float(form.get("Sqft_per_bed"))
        self.Total_Rooms = int(form.get("Total_Rooms"))
        self.is_high_rise = int(form.get("is_high_rise"))
        self.Location = str(form.get("Location", "")).strip()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("rent_form.html", {"request": request})


@app.get("/train")
async def train_model():
    try:
        TrainPipeline().run_pipeline()
        return Response("Training completed successfully!")
    except Exception as e:
        return Response(f"Error occurred: {e}")


@app.post("/")
async def predict(request: Request):
    try:
        form = DataForm(request)
        await form.get_property_data()

        # Compute engineered features
        sqft_per_bed = form.Area_in_sqft / max(form.Beds, 1)
        total_rooms = form.Beds + form.Baths
        
        property_data = PropertyData(
            Area_in_sqft=form.Area_in_sqft,
            Beds=form.Beds,
            Baths=form.Baths,
            Sqft_per_bed=form.Sqft_per_bed,
            Total_Rooms=form.Total_Rooms,
            is_high_rise=form.is_high_rise,
            Location=form.Location or "",
        )

        df = property_data.get_property_input_dataframe()
        predictor = PropertyPredictor()
        prediction = predictor.predict(df, location=form.Location or "")[0]

        return templates.TemplateResponse(
            "rent_form.html",
            {
                "request": request,
                "prediction": f"Estimated Monthly Rent: AED {round(prediction, 2)}"
            }
        )

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)