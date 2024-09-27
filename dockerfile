FROM python:3.11

# Set the working directory
WORKDIR /code

# Copy requirements file
COPY requirements.txt /code/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app /code/app

# Expose the application port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
