# Use an official lightweight Python image
FROM python:3.10-slim

# Avoid Python buffering
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements (for caching layers)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Hugging Face Spaces port
EXPOSE 7860

# Run FastAPI with uvicorn, using api.py as the entrypoint
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
