from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from uvicorn import run as app_run
from fastapi.responses import RedirectResponse

from typing import Optional

from src.constants import APP_HOST, APP_PORT, SAVED_MODEL_FILE_PATH, SAVED_MODEL_DIR_NAME, MLFLOW_EXPERIMENT_NAME
from src.pipeline.prediction_pipeline import PropertyData, PropertyPredictor
from src.pipeline.training_pipeline import TrainPipeline
from contextlib import asynccontextmanager
import os
import mlflow
import threading
import dagshub
import logging

# Configure logging for Render
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Handle DagsHub Authentication
dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    logger.info("DAGSHUB_USER_TOKEN found, configured MLflow authentication")
else:
    logger.warning("DAGSHUB_USER_TOKEN not found. Public repos might work, but private ones will fail.")

try:
    # Use the official DagsHub integration
    dagshub.init(repo_owner='yashbhatewara', repo_name='MLOps_Project_2', mlflow=True)
    logger.info("DagsHub successfully initialized via dagshub.init")
except Exception as e:
    logger.warning(f"DagsHub initialization failed: {e}")
    # Fallback to explicit URI if init fails
    tracking_url = "https://dagshub.com/yashbhatewara/MLOps_Project_2.mlflow"
    mlflow.set_tracking_uri(tracking_url)
    logger.info(f"Fallback tracking URI set to: {tracking_url}")

def download_model():
    if not os.path.exists(SAVED_MODEL_FILE_PATH):
        logger.info(f"Model not found at {SAVED_MODEL_FILE_PATH}. Attempting download from DagsHub...")
        logger.info(f"Current MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        os.makedirs(SAVED_MODEL_DIR_NAME, exist_ok=True)
        try:
            experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            if not experiment:
                logger.warning(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found. Listing all experiments...")
                all_exps = mlflow.search_experiments()
                for exp in all_exps:
                    logger.info(f"Found experiment: {exp.name} (ID: {exp.experiment_id})")
                
                # Try to use the first non-default experiment found
                if len(all_exps) > 0:
                    experiment = all_exps[0]
                    logger.info(f"Falling back to experiment: {experiment.name}")

            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string="attributes.status = 'FINISHED'",
                    order_by=["attributes.start_time DESC"],
                    max_results=5
                )
                for _, run in runs.iterrows():
                    try:
                        logger.info(f"Checking run {run.run_id} for model_package artifact...")
                        mlflow.artifacts.download_artifacts(
                            run_id=run.run_id,
                            artifact_path="model_package/model.pkl",
                            dst_path=SAVED_MODEL_DIR_NAME
                        )
                        # Move file if it's nested (mlflow sometimes creates subfolders)
                        downloaded_file = os.path.join(SAVED_MODEL_DIR_NAME, "model_package", "model.pkl")
                        if os.path.exists(downloaded_file):
                            import shutil
                            shutil.move(downloaded_file, SAVED_MODEL_FILE_PATH)
                            shutil.rmtree(os.path.join(SAVED_MODEL_DIR_NAME, "model_package"))
                        
                        if os.path.exists(SAVED_MODEL_FILE_PATH):
                            logger.info(f"Successfully downloaded model to {SAVED_MODEL_FILE_PATH}")
                            break
                    except Exception as e:
                        logger.error(f"Failed to check run {run.run_id}: {e}")
                        continue
            else:
                logger.error(f"Experiment {MLFLOW_EXPERIMENT_NAME} not found.")
        except Exception as e:
            logger.error(f"Error during model download: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Download model in background to avoid port scan timeout
    thread = threading.Thread(target=download_model)
    thread.start()
    yield

app = FastAPI(lifespan=lifespan)

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
        self.Area_in_sqft = float(form.get("Area_in_sqft", 0))
        self.Beds = int(form.get("Beds", 1))
        self.Baths = int(form.get("Baths", 1))
        self.Sqft_per_bed = float(form.get("Sqft_per_bed", 0))
        self.Total_Rooms = int(form.get("Total_Rooms", 0))
        self.is_high_rise = int(form.get("is_high_rise", 0))
        self.Location = str(form.get("Location", "")).strip()
        self.Type = str(form.get("Type", "")).strip()
        self.Furnishing = str(form.get("Furnishing", "")).strip()


@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
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
        logger.info(f"Received prediction request for location: {form.Location}")

        if not os.path.exists(SAVED_MODEL_FILE_PATH):
            logger.warning("Model file not found. It might be downloading.")
            return templates.TemplateResponse(
                "rent_form.html",
                {"request": request, "error": "Model is still downloading from DagsHub. Please wait a minute and refresh."}
            )

        # Re-compute derived features
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
        logger.info(f"Prediction successful: {prediction}")

        # Render template directly instead of redirecting
        return templates.TemplateResponse(
            "rent_form.html",
            {"request": request, "prediction": prediction}
        )

    except Exception as e:
        import traceback
        logger.error(f"Error during prediction: {e}")
        logger.error(traceback.format_exc())
        return templates.TemplateResponse(
            "rent_form.html",
            {"request": request, "error": f"Error occurred: {str(e)}"}
        )


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
