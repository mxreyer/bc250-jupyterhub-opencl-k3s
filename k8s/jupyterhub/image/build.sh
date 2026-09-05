#!/usr/bin/env bash
# Build the Hub image and load it into k3s's containerd. Run on the host, needs sudo.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=bc250-jupyterhub:latest

sudo docker build -t "$IMAGE" .
sudo docker save "$IMAGE" | sudo k3s ctr images import -

echo
echo "in k3s now:"
sudo k3s ctr images ls | grep bc250-jupyterhub || echo "  (not found - import failed?)"
