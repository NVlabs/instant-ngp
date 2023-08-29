#!/bin/bash

ROOT_DIR=$(dirname ${0})
cd $ROOT_DIR/../..

AWS_REGISTRY=743499434080.dkr.ecr.eu-west-1.amazonaws.com
VERSION=v2.0.0
PROJECT_NAME=3dml-instant-ngp

aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $AWS_REGISTRY

DOCKER_BUILDKIT=1 docker build --build-arg GIT_ACCESS_TOKEN=$GIT_ACCESS_TOKEN -t $PROJECT_NAME:$VERSION -f .devcontainer/Dockerfile .

if ! [ $? -eq 0 ]; then
    echo "Failed to build docker"
    exit $?
fi

docker tag $PROJECT_NAME:$VERSION $AWS_REGISTRY/$PROJECT_NAME:$VERSION
docker push $AWS_REGISTRY/$PROJECT_NAME:$VERSION