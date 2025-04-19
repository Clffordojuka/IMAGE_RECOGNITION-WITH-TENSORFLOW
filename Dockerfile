# Use an official TensorFlow runtime as a parent image
FROM tensorflow/tensorflow:2.15.0

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Set environment variable to avoid Python buffering issues
ENV PYTHONUNBUFFERED=1

# Run your script
CMD ["python", "main.py"]