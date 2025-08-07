#!/bin/bash

echo "⚡ QUICK CI/CD PIPELINE SETUP"
echo "============================"

# Step 1: Create ECR repository using mlops-user
echo ""
echo "📦 Creating ECR repository..."
aws ecr create-repository --repository-name bike-rental-prediction --region us-east-1 2>/dev/null || echo "ECR repository may already exist"

# Step 2: Get ECR repository URI
ECR_URI=$(aws ecr describe-repositories --repository-names bike-rental-prediction --region us-east-1 --query 'repositories[0].repositoryUri' --output text 2>/dev/null)
echo "📋 ECR Repository URI: $ECR_URI"

# Step 3: Display GitHub secrets needed
echo ""
echo "🔐 GITHUB SECRETS NEEDED"
echo "======================="
echo ""
echo "Go to: https://github.com/marcusmayo/neural-networks/settings/secrets/actions"
echo ""
echo "Add these 5 secrets:"

# Get current AWS identity
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown")
echo ""
echo "1. AWS_ACCESS_KEY_ID"
echo "   Value: [Get from: aws iam list-access-keys --user-name mlops-user]"
echo ""
echo "2. AWS_SECRET_ACCESS_KEY" 
echo "   Value: [Your mlops-user secret access key]"
echo ""
echo "3. EC2_HOST"
echo "   Value: [Your EC2 public IP address]"
echo "   Current EC2 instances:"
aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,PublicIpAddress,State.Name]' --output table 2>/dev/null || echo "   Run: aws ec2 describe-instances to get your IP"
echo ""
echo "4. EC2_USER"
echo "   Value: ubuntu"
echo ""
echo "5. EC2_SSH_PRIVATE_KEY"
echo "   Value: [Content of your EC2 private key file]"
echo "   Example: cat ~/.ssh/your-ec2-key.pem"

# Step 4: Check if we can get access keys info
echo ""
echo "📋 Current mlops-user access keys:"
aws iam list-access-keys --user-name mlops-user --query 'AccessKeyMetadata[*].[AccessKeyId,Status]' --output table 2>/dev/null || echo "Could not retrieve access keys info"

# Step 5: Verify mlops-user permissions
echo ""
echo "🔍 Verifying mlops-user permissions..."
echo "✅ EC2 Access:" 
aws ec2 describe-instances --query 'length(Reservations[])' --output text >/dev/null 2>&1 && echo "   Working" || echo "   Failed"

echo "✅ S3 Access:"
aws s3 ls >/dev/null 2>&1 && echo "   Working" || echo "   Failed"

echo "✅ ECR Access:"
aws ecr describe-repositories >/dev/null 2>&1 && echo "   Working" || echo "   Need to add ECR permissions"

# Step 6: Final instructions
echo ""
echo "🎯 FINAL STEPS"
echo "============="
echo ""
echo "1. Add the 5 GitHub secrets listed above"
echo "2. Commit and push your changes:"
echo "   git add ."
echo "   git commit -m 'Add GitHub Actions CI/CD pipeline'"
echo "   git push origin main"
echo ""
echo "3. Monitor the pipeline:"
echo "   https://github.com/marcusmayo/neural-networks/actions"
echo ""
echo "4. Once deployed, your API will be available at:"
echo "   http://[your-ec2-ip]/invocations"
echo "   http://[your-ec2-ip]/health"
echo ""
echo "🚀 Your MLOps pipeline will then:"
echo "   ✅ Test code quality on every push"
echo "   ✅ Build and test Docker images"
echo "   ✅ Deploy to production automatically"
echo "   ✅ Run integration tests"
echo "   ✅ Retrain model weekly (Sundays 2 AM UTC)"
