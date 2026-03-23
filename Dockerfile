FROM python:3.10-slim-bookworm
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y
RUN pip install --no-cache-dir -r requirements.txt


CMD ["python", "app.py"]
