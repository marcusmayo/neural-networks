#!/bin/bash

echo "🔐 REMOVING SECRETS FROM GIT HISTORY"
echo "===================================="
echo ""
echo "GitHub detected AWS credentials in your commit."
echo "We need to remove them from git history."
echo ""

# Step 1: Remove the problematic file
echo "🗑️  Removing file with secrets..."
rm -f quick_pipeline_setup.sh

# Step 2: Check current git status
echo ""
echo "📊 Current git status:"
git status

# Step 3: Reset the problematic commit
echo ""
echo "🔄 Resetting the commit with secrets..."
git reset --soft HEAD~1

# Step 4: Create a clean version of quick_pipeline_setup.sh without secrets
echo ""
echo "📝 Creating clean version of quick_pipeline_setup.sh..."
cat > quick_pipeline_setup.sh << 'EOF'
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
EOF

# Step 5: Stage the clean files
echo ""
echo "📁 Adding clean files..."
git add .

# Step 6: Create a new clean commit
echo ""
echo "💾 Creating clean commit..."
git commit -m "Add GitHub Actions CI/CD pipeline (secrets removed)"

# Step 7: Push the clean version
echo ""
echo "🚀 Pushing clean version..."
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Clean push completed!"
    echo ""
    echo "🔐 IMPORTANT: Your AWS credentials are no longer in git history"
    echo ""
    echo "📊 Next steps:"
    echo "1. Add GitHub secrets manually: https://github.com/marcusmayo/neural-networks/settings/secrets/actions"
    echo "2. Get your actual secret values:"
    echo "   - AWS_ACCESS_KEY_ID: aws iam list-access-keys --user-name mlops-user"
    echo "   - AWS_SECRET_ACCESS_KEY: [from when you created the key]"
    echo "   - EC2_HOST: aws ec2 describe-instances --query 'Reservations[*].Instances[*].PublicIpAddress'"
    echo "   - EC2_USER: ubuntu"
    echo "   - EC2_SSH_PRIVATE_KEY: cat ~/.ssh/your-key.pem"
    echo "3. Monitor pipeline: https://github.com/marcusmayo/neural-networks/actions"
else
    echo ""
    echo "❌ Push failed. Checking status..."
    git status
    echo ""
    echo "🔧 You may need to manually resolve any remaining issues."
fi
