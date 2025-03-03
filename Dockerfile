# Use the official Python 3.13.2 image (if available) from Docker Hub
FROM python:3.13.2-slim

# Set environment variables to avoid Python buffering
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt /app/ 

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app/

# Expose the port your application will use (if applicable)
EXPOSE 8000

# Command to run the Python application (replace with your entry point)
CMD ["python", "langChain_RAG.py"] 
