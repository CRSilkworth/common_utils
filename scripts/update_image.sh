#!/bin/bash
: "${ENVIRONMENT:?ENVIRONMENT must be set (e.g. dev, staging, prd)}"
: "${REGION:?REGION must be set (e.g. asia-northeast1)}"
# Go to the repo root (this works no matter where the script is run from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Extract version from pyproject.toml
VERSION=$(grep '^version =' pyproject.toml | sed -E 's/version = "(.*)"/\1/')
VERSION_TAG="v${VERSION}"
echo "Version: ${VERSION_TAG}"

# Update Terraform tfvars file
TFVARS_PATH="$SCRIPT_DIR/../../../terraform/dev/terraform.tfvars"
if [ -f "$TFVARS_PATH" ]; then
    echo "Updating runner_version_tag in $TFVARS_PATH"
    sed -i.bak -E "s/^runner_version_tag *= *\"v[0-9]+\.[0-9]+\.[0-9]+\"/runner_version_tag = \"${VERSION_TAG}\"/" "$TFVARS_PATH"
    rm -f "${TFVARS_PATH}.bak"
else
    echo "Warning: Terraform tfvars file not found at $TFVARS_PATH"
fi

# Make normalized temporary pyproject (for caching)
cp pyproject.toml pyproject.tmp.toml
sed -i.bak -E 's/^version = ".*"/version = "0.0.0"/' pyproject.tmp.toml
rm -f pyproject.tmp.toml.bak

# Build image tag
IMAGE_TAG="${REGION}-docker.pkg.dev/carbon-432311/carbon/runner:${VERSION_TAG}-${ENVIRONMENT}"

echo "Building Docker image: $IMAGE_TAG"
docker build \
    --build-arg PYPROJECT_FILE=pyproject.tmp.toml \
    -f Dockerfile.runner \
    -t "$IMAGE_TAG" .

# Clean temp file
rm pyproject.tmp.toml

# Conditional push or load
if [ "$ENVIRONMENT" = "prd" ]; then
    echo "Environment is prd — pushing to registry..."
    docker push "$IMAGE_TAG"
else
    echo "Environment is $ENVIRONMENT — loading into Minikube..."
    minikube image load "$IMAGE_TAG"
fi

