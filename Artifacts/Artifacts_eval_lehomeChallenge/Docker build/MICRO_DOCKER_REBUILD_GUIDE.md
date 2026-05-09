# LeHome Challenge 2026 - Micro Docker Rebuild Artifact

This artifact documents the evaluator-facing micro rebuild process discussed in the audit thread.

## Integrity Scope

- Base image: `nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell` (original submitted image).
- Checkpoint integrity: unchanged (`model.safetensors` is not retrained or replaced).
- Targeted non-checkpoint patch scope:
  - Container-side wrapper fix: `policy.py` only (normalization + un-normalization flow).
  - Container-side metadata add: `meta/` only (normalization statistics).
  - Host-side compatibility patch: `warp-lang==1.11.1` plus eval adapter updates from `apply_host_updates.sh`.

## Why This Micro Rebuild Exists

The original container wrapper bypassed LeRobot pre/post processors and sent unnormalized observations directly to the policy while returning un-denormalized actions.  
This micro rebuild fixes that wrapper behavior without changing model weights.

## Required Build Context

Run from:

`lehome_workspace/lehome-challenge/dummy_docker_policy/`

Expected files in context:

- `Dockerfile.patch`
- `policy.py` (patched inference wrapper)
- `meta/` (dataset metadata/statistics used by pre/post processors)
- `pretrained_model/` (contains `config.json`, `policy_preprocessor.json`, `policy_postprocessor.json` and matching `*.safetensors` required by the new `policy.py` to instantiate the LeRobot `PolicyProcessorPipeline`; the base image only ships `model.safetensors`)

## Build Commands

```bash
docker login -u nninspaceexp --password-stdin <<< "<REDACTED_DOCKER_PAT>"
docker build -t nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell_patched-Updated -f Dockerfile.patch .
```

## Push Command

```bash
docker push nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell_patched-Updated
```

## Evaluator Runtime Tag

Use this patched image when starting policy containers:

`nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell_patched-Updated`

The host-side evaluation command (matches V2 README Step 8):

```bash
mkdir -p /data/warp_cache
WARP_CACHE_PATH=/data/warp_cache OMNI_KIT_ACCEPT_EULA=yes PYTHONUNBUFFERED=1 \
xvfb-run -a python -u -m scripts.eval --policy_type docker --docker_url http://localhost:8081 ...
```

When starting the policy container, pass `-e HF_HOME=/tmp` so the in-container LeRobot dataset metadata cache is writable for the non-root `user`:

```bash
docker run --rm --gpus all -e HF_HOME=/tmp -p 8081:8080 \
    nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell_patched-Updated
```

## Notes

- This is intentionally a surgical patch layer, not a full rebuild.
- If push fails, the most common cause is Docker Hub credentials lacking write scope.
