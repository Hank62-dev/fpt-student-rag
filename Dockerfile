FROM python:3.11-slim

WORKDIR /app

# Cài dependencies hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Tạo thư mục data và indexes
RUN mkdir -p backend/data backend/indexes

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]