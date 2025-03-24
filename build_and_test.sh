#!/bin/bash
set -e

# Build the Docker image
docker build -t rgt-api:latest .

# Run the container
docker run -d -p 8000:8000 --name rgt-api-container rgt-api:latest

echo "Container started. API should be available at http://localhost:8000"
echo "To check the logs: docker logs rgt-api-container"
echo "To stop the container: docker stop rgt-api-container && docker rm rgt-api-container"