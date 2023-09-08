#!/bin/bash

set -e

# Default values
VERSION=""

# Function to display usage instructions
usage() {
    echo "Usage: $0 -v VERSION"
    exit 1
}

# Parse command line options
while getopts ":d:v:c:" opt; do
    case $opt in
        v)
            VERSION="$OPTARG"
            ;;
        *)
            usage
            ;;
    esac
done

ROOT_DIR=$(dirname ${0})
cd $ROOT_DIR/../..

AWS_REGISTRY=743499434080.dkr.ecr.eu-west-1.amazonaws.com
PROJECT_NAME=3dml-instant-ngp

aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $AWS_REGISTRY

DOCKER_BUILDKIT=1 docker build --build-arg APP_ENV=build --build-arg GIT_ACCESS_TOKEN=$GIT_ACCESS_TOKEN -t $PROJECT_NAME:$VERSION -f .devcontainer/Dockerfile .

docker tag $PROJECT_NAME:$VERSION $AWS_REGISTRY/$PROJECT_NAME:$VERSION
docker push $AWS_REGISTRY/$PROJECT_NAME:$VERSION