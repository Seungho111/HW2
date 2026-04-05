# Use a slim python base image
FROM python:3.10-slim

# Prevent python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Only copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model into the image cache to drastically reduce container startup time
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B-Chat'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-0.5B-Chat')"

# Copy remaining source code
COPY . .

# Run as non-root user for better security (MLOps Best Practice)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
