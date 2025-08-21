#"""
#chmod +x push-to-gcp.sh
#./push-to-gcp.sh \
#  --project owl-3d-462416 \
#  --image cut3e-pr-42 \
#  --tag latest \
#  --path .
#"""

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --project PROJECT_ID --image IMAGE_NAME --tag TAG [--path CONTEXT]
Options:
  --project   GCP project ID (e.g. owl-api-staging)
  --image     Image name (e.g. cut3e-pr-42)
  --tag       Tag for this build (e.g. latest)
  --path      Build context directory (default: current directory)
EOF
  exit 1
}

BUILD_PATH="."
while [[ $# -gt 0 ]]; do
  case $1 in
    --project) PROJECT="$2"; shift 2 ;;
    --image)   IMAGE="$2";   shift 2 ;;
    --tag)     TAG="$2";     shift 2 ;;
    --path)    BUILD_PATH="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -z "${PROJECT:-}" || -z "${IMAGE:-}" || -z "${TAG:-}" ]] && usage

REPO="us-central1-docker.pkg.dev/$PROJECT/skypilot"
FULL_IMAGE="$REPO/$IMAGE:$TAG"

echo "🔑 1. Authenticate & select project"
gcloud auth login --quiet
gcloud config set project "$PROJECT"

echo "🚀 2. Enable Artifact Registry API"
gcloud services enable artifactregistry.googleapis.com --quiet

echo "🐳 3. Configure Docker credential helper"
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

echo "🏗 4. Build image: $FULL_IMAGE"
docker build -t "$FULL_IMAGE" "$BUILD_PATH" --no-cache

echo "📤 5. Push image"
docker push "$FULL_IMAGE"

echo "✅ Done! You can pull with:"
echo "    docker pull $FULL_IMAGE"