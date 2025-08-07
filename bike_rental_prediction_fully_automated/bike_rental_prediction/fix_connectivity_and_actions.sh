#!/bin/bash

echo "🔧 FIXING NETWORK CONNECTIVITY AND GITHUB ACTIONS"
echo "=================================================="

# Step 1: Check what's actually running and on what ports
echo ""
echo "🔍 Checking running services..."
echo "Network connections:"
netstat -tlnp | grep -E ':(1234|5000|80)'

echo ""
echo "Running MLflow processes:"
ps aux | grep mlflow

# Step 2: Fix MLflow services
echo ""
echo "🔄 Restarting MLflow services properly..."

# Kill existing MLflow processes
pkill -f "mlflow ui"
pkill -f "mlflow models serve"

# Wait a moment
sleep 3

# Start MLflow UI on all interfaces
echo "🚀 Starting MLflow UI on 0.0.0.0:5000..."
nohup mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:///home/ubuntu/neural-networks/bike_rental_prediction_fully_automated/bike_rental_prediction/mlruns > mlflow-ui.log 2>&1 &

# Wait for UI to start
sleep 5

# Start model serving
echo "🚀 Starting model serving on 0.0.0.0:1234..."
cd ~/neural-networks/bike_rental_prediction_fully_automated/bike_rental_prediction
nohup mlflow models serve -m "models:/BikeRentalModel/latest" --host 0.0.0.0 -p 1234 --no-conda > serve.log 2>&1 &

# Step 3: Wait and test local connectivity
echo ""
echo "⏳ Waiting 30 seconds for services to fully start..."
sleep 30

echo ""
echo "🧪 Testing local connectivity..."

# Test localhost connections
echo "Testing MLflow UI (localhost:5000):"
curl -s -m 5 http://localhost:5000 >/dev/null && echo "✅ MLflow UI accessible locally" || echo "❌ MLflow UI not accessible locally"

echo "Testing model serving (localhost:1234):"
curl -s -m 5 http://localhost:1234/health && echo "✅ Model serving accessible locally" || echo "❌ Model serving not accessible locally"

# Step 4: Check firewall
echo ""
echo "🛡️ Checking firewall status..."
sudo ufw status || echo "UFW not active"

# Step 5: Fix GitHub Actions
echo ""
echo "📁 Fixing GitHub Actions workflow..."

# Check if we're in the right directory
pwd

# Create/update workflow file with correct path
mkdir -p .github/workflows

cat > .github/workflows/mlops-pipeline.yml << 'EOF'
name: MLOps CI/CD - Bike Rental Prediction

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:  # Manual trigger

env:
  DOCKER_IMAGE: bike-rental-prediction
  AWS_REGION: us-east-1
  ECR_REPOSITORY: bike-rental-prediction

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: bike_rental_prediction_fully_automated/bike_rental_prediction
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Test model training
      run: |
        python src/train.py

  build-docker:
    runs-on: ubuntu-latest
    needs: test
    defaults:
      run:
        working-directory: bike_rental_prediction_fully_automated/bike_rental_prediction
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
          
    - name: Build Docker image
      run: docker build -t ${{ env.DOCKER_IMAGE }} .

  deploy:
    runs-on: ubuntu-latest
    needs: [test, build-docker]
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
        
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v2
      
    - name: Build and push Docker image to ECR
      working-directory: bike_rental_prediction_fully_automated/bike_rental_prediction
      run: |
        docker build -t $ECR_REPOSITORY .
        docker tag $ECR_REPOSITORY:latest ${{ steps.login-ecr.outputs.registry }}/$ECR_REPOSITORY:latest
        docker push ${{ steps.login-ecr.outputs.registry }}/$ECR_REPOSITORY:latest
        
    - name: Deploy to EC2
      env:
        PRIVATE_KEY: ${{ secrets.EC2_SSH_PRIVATE_KEY }}
        HOST: ${{ secrets.EC2_HOST }}
        USER: ${{ secrets.EC2_USER }}
      run: |
        echo "$PRIVATE_KEY" > private_key.pem
        chmod 600 private_key.pem
        
        ssh -i private_key.pem -o StrictHostKeyChecking=no $USER@$HOST << 'EOF'
          docker stop bike-rental-api || true
          docker rm bike-rental-api || true
          
          aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${{ steps.login-ecr.outputs.registry }}
          
          docker pull ${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:latest
          docker run -d --name bike-rental-api -p 80:1234 --restart unless-stopped \
            ${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:latest
        EOF
        
        rm -f private_key.pem
EOF

echo "✅ GitHub Actions workflow updated"

# Step 6: Trigger GitHub Actions
echo ""
echo "🚀 Triggering GitHub Actions..."

# Add workflow file to git
git add .github/workflows/mlops-pipeline.yml

# Check git status
git status

# Create README if it doesn't exist
if [ ! -f README.md ]; then
    echo "# MLOps Bike Rental Prediction Pipeline" > README.md
fi

# Add trigger comment to README
echo "" >> README.md
echo "<!-- Triggered at $(date) -->" >> README.md

# Commit and push
git add .
git commit -m "Fix GitHub Actions workflow and trigger pipeline"
git push origin main

echo ""
echo "📊 CURRENT STATUS:"
echo "=================="
echo ""

# Check what's running now
echo "🔄 Running services:"
netstat -tlnp | grep -E ':(1234|5000)' || echo "No services on ports 1234 or 5000"

echo ""
echo "📋 Service logs (last 10 lines):"
echo "MLflow UI log:"
tail -10 mlflow-ui.log 2>/dev/null || echo "No MLflow UI log found"

echo ""
echo "Model serving log:"
tail -10 serve.log 2>/dev/null || echo "No serving log found"

echo ""
echo "🎯 NEXT STEPS:"
echo "=============="
echo ""
echo "1. ✅ GitHub Actions should now be triggered"
echo "   Monitor: https://github.com/marcusmayo/machine-learning-portfolio/actions"
echo ""
echo "2. 🔧 Test local services:"
echo "   MLflow UI: curl http://localhost:5000"
echo "   Model API: curl http://localhost:1234/health"
echo ""
echo "3. 🌐 Once services are running, test public access:"
echo "   MLflow UI: http://18.233.252.250:5000"
echo "   Model API: http://18.233.252.250:1234/health"
echo ""
echo "4. 📊 If still not accessible, check EC2 security group allows:"
echo "   - Port 5000 (MLflow UI)"
echo "   - Port 1234 (Model serving)"
echo "   - Port 80 (Production API)"
