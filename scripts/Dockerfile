# Use a lightweight python base image
FROM python:3.10-slim

# Install necessary system dependencies (if any are needed by torch/transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create a non-root user for HuggingFace Spaces
RUN useradd -m -u 1000 user
USER user

# Set environment variables for the user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Move working directory to user home
WORKDIR $HOME/app

# Copy the rest of the application code with user ownership
COPY --chown=user . $HOME/app

# Expose port 7860
EXPOSE 7860

# Run FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
