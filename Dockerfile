# Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl .
COPY run_model.py .

CMD ["python", "run_model.py"]
