# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV IN_DOCKER=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if needed) - keep commented unless necessary
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Install pip requirements first
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# --- Permissions Fix for Render.com / Non-Root Execution ---
# Explicitly create directories the app needs to write to.
# This ensures they exist even if not copied fully or if mount points are used.
RUN mkdir -p /app/recipes/json /app/recipes/html /app/rdi /app/diets

# Create the non-root user and group *before* changing ownership
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Change ownership of the entire /app directory and its contents
# to the new non-root user. This grants write permissions.
RUN chown -R appuser:appuser /app

# Switch to the non-root user *after* ownership is changed
USER appuser
# --- End Permissions Fix ---

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app-web.py", "--webui"]