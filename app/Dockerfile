# Use a smaller Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake gcc g++ make \
    libsm6 libxext6 libgl1-mesa-glx \
    libboost-all-dev ffmpeg python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first to leverage caching
COPY requirements.txt /app/

# Upgrade pip
RUN pip install --upgrade pip 

# Install precompiled dlib instead of building from source
RUN pip install --no-cache-dir dlib

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt 

# Copy the rest of the project files
COPY . /app/

# Ensure necessary directories exist
RUN mkdir -p /app/uploaded_videos

# Expose Flask app port
EXPOSE 5000

# Set Flask environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Run the Flask app
CMD ["python", "app.py"]
