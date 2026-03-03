# lightweight Python image
FROM python:3.10-slim

# working directory
WORKDIR /app

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Ensure src is on Python path
ENV PYTHONPATH=/app

# Default command
ENTRYPOINT ["python", "-m", "src.main"]