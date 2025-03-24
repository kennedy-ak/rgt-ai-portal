#!/bin/bash
set -e

# Configuration
AWS_REGION="us-east-1"  # Change to your preferred region
ECR_REPOSITORY_NAME="rgt-api"

# Login to AWS ECR
echo "Logging in to Amazon ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com

# Create ECR repository if it doesn't exist
echo "Creating ECR repository if it doesn't exist..."
aws ecr describe-repositories --repository-names $ECR_REPOSITORY_NAME --region $AWS_REGION || \
    aws ecr create-repository --repository-name $ECR_REPOSITORY_NAME --region $AWS_REGION

# Get the ECR repository URI
ECR_REPOSITORY_URI=$(aws ecr describe-repositories --repository-names $ECR_REPOSITORY_NAME --region $AWS_REGION --query 'repositories[0].repositoryUri' --output text)

# Tag the Docker image
echo "Tagging Docker image..."
docker tag rgt-api:latest $ECR_REPOSITORY_URI:latest

# Push the Docker image to ECR
echo "Pushing Docker image to ECR..."
docker push $ECR_REPOSITORY_URI:latest

echo "Docker image pushed successfully to: $ECR_REPOSITORY_URI:latest"
echo "Use this URI in your AWS App Runner configuration."