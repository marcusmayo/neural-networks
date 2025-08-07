#!/bin/bash

echo "🔧 Starting deployment on EC2..."

# Stop ALL containers
echo "Stopping all Docker containers..."
docker stop $(docker ps -aq) 2>/dev/null || true
docker rm $(docker ps -aq) 2>/dev/null || true

# Remove ALL old images to force fresh pull
echo "Removing old images..."
docker rmi $(docker images -q) 2>/dev/null || true
docker system prune -af --volumes

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 453553127570.dkr.ecr.us-east-1.amazonaws.com

# Pull latest image (will be forced to download fresh)
echo "Pulling latest image..."
docker pull 453553127570.dkr.ecr.us-east-1.amazonaws.com/bike-rental-prediction:latest

# Run new container
echo "Starting new container..."
docker run -d \
  --name bike-rental-api \
  -p 80:1234 \
  --restart unless-stopped \
  453553127570.dkr.ecr.us-east-1.amazonaws.com/bike-rental-prediction:latest

# Wait for container to fully start
echo "Waiting for container to start..."
sleep 10

# Verify container is running
docker ps

# Check container logs
echo "Container logs:"
docker logs bike-rental-api --tail 50

# Test all endpoints
echo "Testing endpoints..."
echo "Root endpoint:"
curl -v http://localhost/
echo -e "\n\nHealth endpoint:"
curl -v http://localhost/health
echo -e "\n\nPredict endpoint test:"
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3]}'

echo -e "\n✅ Deployment complete!"
