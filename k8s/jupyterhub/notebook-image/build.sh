#!/usr/bin/env bash
# Build the BC-250 notebook image and load it into k3s's containerd store.
# Run on the host. Needs sudo (for docker and for k3s ctr).
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=bc250-notebook:latest
# Fedora release of the base image. Defaults to the Dockerfile's default (45);
# FEDORA_VERSION=44 ./build.sh gets the previous one back.
BUILD_ARGS=()
[ -n "${FEDORA_VERSION:-}" ] && BUILD_ARGS+=(--build-arg "FEDORA_VERSION=$FEDORA_VERSION")

sudo docker build "${BUILD_ARGS[@]}" -t "$IMAGE" .

# k3s runs its OWN containerd, separate from the Docker daemon. An image built
# by Docker is not visible to k3s until it's imported. `docker save | k3s ctr
# images import -` streams it across into k3s's k8s.io namespace.
sudo docker save "$IMAGE" | sudo k3s ctr images import -

echo
echo "in k3s now:"
sudo k3s ctr images ls | grep bc250-notebook || echo "  (not found - import failed?)"
