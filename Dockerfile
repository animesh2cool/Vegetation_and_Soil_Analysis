FROM python:3.10-slim

# Avoid Python buffering issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (OpenCV + PIL need these)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Hugging Face requires port 7860
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
