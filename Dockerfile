# Use an official Python runtime as a parent image
# Choose a version compatible with your project (e.g., 3.11)
# Use the slim variant for a smaller image size
FROM python:3.11-slim

# Set environment variables
# Prevents Python from buffering stdout/stderr, making logs appear immediately
ENV PYTHONUNBUFFERED=1
# Set Flask environment (optional, 'production' is safer than default 'development')
ENV FLASK_ENV=production
# Indicate we're running within Docker (optional, might be useful for conditional logic)
ENV IN_DOCKER=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for pandas/numpy compilation)
# Uncomment and add packages if build fails (common ones: build-essential, libatlas-base-dev)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Install pip requirements first to leverage Docker cache
# Copy only the requirements file
COPY requirements.txt requirements.txt

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the working directory
COPY . .

# Create necessary directories for output within the container image
# Ensure the application has a place to write reports and history
# RUN mkdir -p /app/recipes/json /app/recipes/html /app/rdi /app/diets
# Note: COPY . . above should create these if they exist locally,
# but explicitly creating them ensures they exist if source dirs are empty or missing.
# However, if directories are definitely part of the COPY, this RUN might be redundant.
# Let's ensure the base output dir exists at least.
RUN mkdir -p /app/recipes

# --- Security Best Practice: Run as non-root user ---
# Create a non-root user and group
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser
# Optional: Change ownership of the app directory if needed
# RUN chown -R appuser:appuser /app
# Switch to the non-root user
USER appuser
# --- End non-root user setup ---

# Expose the port the app runs on (Flask default is 5000, check your run_webui function)
EXPOSE 5000

# Define the command to run the application
# Use the exec form to properly handle signals
CMD ["python", "app-web.py", "--webui"]