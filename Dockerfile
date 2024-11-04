# Start with the official Python 3.10 image
FROM python:3.10

# Set the working directory
WORKDIR /code

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install TensorFlow
RUN pip install tensorflow==2.18.0

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install Gunicorn
RUN pip install gunicorn

# Copy the application code
COPY ./app /code/app

# Command to run the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:80", "app.main:app"]