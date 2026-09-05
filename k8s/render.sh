#!/usr/bin/env bash
# Render k8s/**/<name>.yaml.tmpl -> k8s/**/<name>.yaml, substituting the
# host-specific values from k8s/env.local. Needs `envsubst` (gettext).
#
#   cp k8s/env.example k8s/env.local && $EDITOR k8s/env.local
#   ./k8s/render.sh
#   sudo kubectl apply -f k8s/device-plugin/ -f k8s/jupyterhub/   # etc.
#
# The rendered *.yaml are gitignored; only the *.yaml.tmpl and env.example are
# committed. Re-run this after editing env.local or any template.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f env.local ]]; then
  echo "k8s/env.local not found." >&2
  echo "  cp k8s/env.example k8s/env.local && \$EDITOR k8s/env.local" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
. ./env.local
set +a

: "${NODE_LAN_IP:?set it in k8s/env.local}"
: "${TS_HOSTNAME:?set it in k8s/env.local}"
: "${TS_TAILNET:?set it in k8s/env.local}"
: "${TS_TAG:?set it in k8s/env.local}"
: "${HUB_ADMIN:?set it in k8s/env.local}"

# Explicit allow-list: envsubst leaves every other $VAR / ${VAR} untouched
# (e.g. ${TS_CERT_DOMAIN} in 35-tailscale-serve.yaml, os.environ[...] in the
# hub config).
subst='${NODE_LAN_IP} ${TS_HOSTNAME} ${TS_TAILNET} ${TS_TAG} ${HUB_ADMIN}'

n=0
while IFS= read -r -d '' tmpl; do
  out="${tmpl%.tmpl}"
  envsubst "$subst" < "$tmpl" > "$out"
  echo "rendered ${out#./}"
  n=$((n + 1))
done < <(find . -name '*.yaml.tmpl' -print0 | sort -z)

[[ $n -gt 0 ]] || { echo "no *.yaml.tmpl found under k8s/" >&2; exit 1; }
echo "$n file(s) rendered."
