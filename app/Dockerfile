FROM python:3.9-slim

# Set the working directory
WORKDIR /app

RUN mkdir logs

# Copy the API dependencies file 
COPY requirements-api.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the application code into the container
COPY api_server.py .

# Expose the port the API will run on
EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
