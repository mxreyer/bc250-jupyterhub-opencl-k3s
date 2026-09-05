# The Hub image.
#
# The upstream jupyterhub/jupyterhub image already bundles jupyterhub, the
# configurable-http-proxy (Node), and DummyAuthenticator. It does NOT include a
# Kubernetes spawner, so we add jupyterhub-kubespawner. That's the whole image.
#
# Running the proxy in-process (the Hub's default) keeps this to ONE pod - no
# separate proxy Deployment to wire up. Fine for a single-node homelab hub.
FROM quay.io/jupyterhub/jupyterhub:5

RUN python3 -m pip install --no-cache-dir \
        "jupyterhub-kubespawner==7.0.0" \
        "jupyterhub-idle-culler==1.4.0" \
        "jupyterhub-nativeauthenticator>=1.2,<2"
# jupyterhub-idle-culler runs as a Hub-managed service (configured in
# jupyterhub_config.py) that shuts down user servers idle past a timeout, so an
# unused notebook stops holding the GPU device and its memory.
#
# jupyterhub-nativeauthenticator gives each user their own password: self
# sign-up at /hub/signup, admin approval at /hub/authorize, self-service
# password change at /hub/change-password. Credentials (bcrypt) are stored in
# the Hub DB, which lives on the hub-state PVC. It auto-registers its own
# handlers and templates when set as the authenticator_class.
