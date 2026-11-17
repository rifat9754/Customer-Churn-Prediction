
FROM python:3.14-slim

WORKDIR /app

# Install API dependencies
COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

# Copy application
COPY api/ ./api/

# Copy trained ML models
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
