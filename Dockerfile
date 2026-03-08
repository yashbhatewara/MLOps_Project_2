FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y
RUN pip install --no-cache-dir -r requirements.txt

# Use Gunicorn with Uvicorn workers for FastAPI compatibility
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker app:app
