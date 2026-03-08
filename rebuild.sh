#!/usr/bin/env bash
# RevGate full rebuild -- по Helena конвенцията.

set -e

echo ""
echo "=== FULL REBUILD ==="
cd ~/revgate

docker stop revgate-dev 2>/dev/null || true
docker rm revgate-dev 2>/dev/null || true
docker rmi revgate:development 2>/dev/null || true
docker buildx prune -af

docker build \
    --no-cache \
    -f Dockerfile \
    --target development \
    -t revgate:development \
    .

# Ensure network exists
docker network create revgate-dev-network 2>/dev/null || echo "Network exists"

docker compose -f docker-compose-development.yaml up -d revgate

echo ""
echo "=== BUILD COMPLETE ==="
docker ps | grep revgate-dev
