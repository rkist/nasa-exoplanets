# Use lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional but good practice)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the backend source code
COPY api/ /app/

# Expose the Flask port
EXPOSE 5000

# Start the app with Gunicorn
CMD ["gunicorn", "app:app", "--workers", "4", "--bind", "0.0.0.0:3000", "--timeout", "120"]
