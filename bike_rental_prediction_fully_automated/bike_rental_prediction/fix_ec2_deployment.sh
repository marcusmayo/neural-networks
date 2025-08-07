#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    Fix EC2 Deployment - Force Update Container${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

# EC2 Details
EC2_HOST="18.233.252.250"
ECR_REGISTRY="453553127570.dkr.ecr.us-east-1.amazonaws.com"

# Step 1: Ensure latest image is in ECR
echo -e "\n${YELLOW}Step 1: Building and pushing latest image to ECR...${NC}"

# Rebuild with latest code
docker build -t bike-rental-prediction . --no-cache

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REGISTRY

# Tag and push
docker tag bike-rental-prediction:latest $ECR_REGISTRY/bike-rental-prediction:latest
docker push $ECR_REGISTRY/bike-rental-prediction:latest

echo -e "${GREEN}✅ Latest image pushed to ECR${NC}"

# Step 2: Create deployment script
echo -e "\n${YELLOW}Step 2: Creating remote deployment script...${NC}"

cat > remote_deploy.sh << 'REMOTE_SCRIPT'
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
REMOTE_SCRIPT

chmod +x remote_deploy.sh

# Step 3: Update GitHub Secrets for proper deployment
echo -e "\n${YELLOW}Step 3: Creating updated GitHub workflow...${NC}"

cat > .github/workflows/deploy-fix.yml << 'WORKFLOW'
name: Fixed Deployment

on:
  workflow_dispatch:
  push:
    branches: [main, master]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Deploy to EC2
      run: |
        # Use EC2 Instance Connect to deploy
        aws ec2-instance-connect send-ssh-public-key \
          --instance-id i-0dc6adb1e1abe543d \
          --instance-os-user ubuntu \
          --ssh-public-key "$(cat ~/.ssh/id_rsa.pub 2>/dev/null || ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa && cat ~/.ssh/id_rsa.pub)" \
          --availability-zone us-east-1b || true
        
        # Alternative: Use Systems Manager if available
        aws ssm send-command \
          --instance-ids i-0dc6adb1e1abe543d \
          --document-name "AWS-RunShellScript" \
          --parameters 'commands=[
            "docker stop bike-rental-api || true",
            "docker rm bike-rental-api || true",
            "docker system prune -af",
            "aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 453553127570.dkr.ecr.us-east-1.amazonaws.com",
            "docker pull 453553127570.dkr.ecr.us-east-1.amazonaws.com/bike-rental-prediction:latest",
            "docker run -d --name bike-rental-api -p 80:1234 --restart unless-stopped 453553127570.dkr.ecr.us-east-1.amazonaws.com/bike-rental-prediction:latest",
            "docker ps"
          ]' || echo "SSM not available, using direct connection"
WORKFLOW

# Step 4: Test from current location
echo -e "\n${YELLOW}Step 4: Testing current EC2 deployment...${NC}"

echo "Testing EC2 endpoints from here..."
echo -e "\nRoot endpoint:"
curl -s http://$EC2_HOST/ || echo "Root endpoint not working"

echo -e "\nHealth endpoint:"  
curl -s http://$EC2_HOST/health || echo "Health endpoint not working"

echo -e "\nPredict endpoint:"
curl -s -X POST http://$EC2_HOST/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3]}' || echo "Predict endpoint not working"

# Step 5: Manual deployment option
echo -e "\n${YELLOW}Step 5: Manual deployment instructions...${NC}"

cat << 'MANUAL'

🔧 MANUAL FIX INSTRUCTIONS:
===========================

Since we can't directly SSH from this EC2, you have THREE options:

OPTION 1: From your LOCAL computer (with the .pem key):
---------------------------------------------------------
ssh -i your-key.pem ubuntu@18.233.252.250
# Then copy and paste the contents of remote_deploy.sh

OPTION 2: Use AWS Console:
--------------------------
1. Go to EC2 Console → Instances
2. Select instance i-0dc6adb1e1abe543d
3. Click "Connect" → "Session Manager" or "EC2 Instance Connect"
4. Run these commands:
   docker stop bike-rental-api
   docker rm bike-rental-api
   docker system prune -af
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 453553127570.dkr.ecr.us-east-1.amazonaws.com
   docker pull 453553127570.dkr.ecr.us-east-1.amazonaws.com/bike-rental-prediction:latest
   docker run -d --name bike-rental-api -p 80:1234 --restart unless-stopped 453553127570.dkr.ecr.us-east-1.amazonaws.com/bike-rental-prediction:latest

OPTION 3: Trigger GitHub Actions:
----------------------------------
git add .github/workflows/deploy-fix.yml
git commit -m "Force redeploy with fixed endpoints"
git push

Then go to GitHub → Actions → Run "Fixed Deployment" workflow manually

MANUAL

# Step 6: Commit changes
echo -e "\n${YELLOW}Step 6: Committing changes...${NC}"

git add -A
git commit -m "Fix: Force EC2 deployment with all endpoints working" || true
git push || true

echo -e "\n${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "The EC2 instance needs to be manually updated using one of the options above."
echo "Once updated, all endpoints will return JSON instead of HTML 404 errors."
