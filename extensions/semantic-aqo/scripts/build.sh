#!/bin/bash
echo "Building ARTIFACTS image for semantic-aqo..."

set -a
[ -f .env ] && . .env
set +a

ARTIFACT_IMAGE="${DOCKER_HUB_USERNAME}/semantic-aqo-artifacts:1.0-pg${POSTGRES_VERSION}"

docker build \
  --build-arg POSTGRES_VERSION=${POSTGRES_VERSION} \
  --target artifacts \
  -t $ARTIFACT_IMAGE \
  -f ../docker/Dockerfile.deploy .

echo "Done. Pushing to Docker Hub..."
docker push $ARTIFACT_IMAGE

echo "Artifact image '$ARTIFACT_IMAGE' is now available for delivery."