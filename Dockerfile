FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PROJECT=customer-segmentation-dashboard
CMD ["sh", "-c", "uvicorn portfolio_core.docker:app --host 0.0.0.0 --port 8000"]
