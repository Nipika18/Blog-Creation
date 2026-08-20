# Use an official Python image as a parent image
FROM python:3.10-slim

# Install Node.js (for building the React frontend)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Set the working directory to /app
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Build the React frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Install Python dependencies for the backend
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7860

# Expose the port Hugging Face Spaces uses
EXPOSE 7860

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
