#!/usr/bin/env bash
# LeHome policy Docker image: W&B download (optional) + docker build (+ optional HF Spaces registry push).
# Intended to run on the GPU VM (Docker, checkpoints, wandb).
#
# Blackwell (RTX 50-series, sm_120): Dockerfile.submission uses pytorch/pytorch:*-cuda12.8-*-runtime.
# It then pins torch/torchvision/torchaudio to 2.7.1 cu128 wheels, which keeps lerobot==0.4.3
# in its declared torch<2.8 compatibility range while preserving Blackwell CUDA support.
# Rebuild and tag for registry, e.g.:
#   ./build_docker_submission.sh --skip-download --image-tag nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
#
# Usage examples:
#   export WANDB_API_KEY=...   # or: wandb login ; or source a .env containing WANDB_API_KEY
#   ./build_docker_submission.sh --wb-artifact "my-run-checkpoint:v3"
#
#   ./build_docker_submission.sh --skip-download --image-tag lehome-policy:local
#
#   source .env   # HF_TOKEN or HUGGING_FACE_HUB_TOKEN + WANDB_API_KEY as needed
#   ./build_docker_submission.sh --wb-artifact "model:v0" --push \
#     --hf-user YOUR_HF_USER --hf-space YOUR_DOCKER_SPACE --hf-remote-tag v1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="${SCRIPT_DIR}"
DOCKERFILE="Dockerfile.submission"
WB_PROJECT="lehome_challenge"
WB_ARTIFACT=""
PRETRAINED_DIR_NAME="pretrained_model"
SKIP_DOWNLOAD="0"
IMAGE_TAG="lehome-policy:local"
DOCKER_BUILD_EXTRA=()
INIT_REQUIREMENTS="0"
ENV_FILE=""
PUSH="0"
HF_REGISTRY="${HF_REGISTRY:-registry.hf.space}"
HF_USER=""
HF_SPACE=""
HF_REMOTE_TAG="v1"
HF_IMAGE_REF=""

load_env_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    echo "Sourcing environment file: $f"
    set -a
    # shellcheck disable=SC1090
    source "$f"
    set +a
  fi
}

usage() {
  cat <<'EOF'
LeHome dummy_docker_policy — one-shot W&B download, docker build, optional HF registry push.

Examples:
  export WANDB_API_KEY=...   # or wandb login, or put keys in .env
  ./build_docker_submission.sh --wb-artifact "my-checkpoint:v0"
  ./build_docker_submission.sh --skip-download --image-tag lehome-policy:local
  ./build_docker_submission.sh --skip-download --image-tag nninspaceexp/lehome_silicon-optimists:FullDP-v2_Blackwell
  ./build_docker_submission.sh --wb-artifact "model:v0" --push --hf-user USER --hf-space SPACE --hf-remote-tag v1

Options:
  --policy-dir DIR          Build context (default: this script's directory)
                            Auto-creates requirements.txt from the template and stages lerobot_policy_dino from
                            ../../lerobot_policy_dino when missing (Dockerfile COPY steps).
  --dockerfile FILE         Dockerfile name (default: Dockerfile.submission)
  --wb-project NAME         W&B project path passed to download_wandb_model.py (default: lehome_challenge).
                            Use entity/project form if your artifact lives under a team entity.
  --wb-artifact NAME        Required unless --skip-download (e.g. model-runid:v0)
  --pretrained-subdir NAME  Directory under policy-dir for weights (default: pretrained_model)
  --skip-download           Do not call download_wandb_model.py (weights must already exist)
  --image-tag TAG           Local docker image name:tag for docker build -t (default: lehome-policy:local)
  --no-cache                Pass --no-cache to docker build
  --env-file PATH           Source this file (set -a / set +a) before wandb/docker login; default tries:
                            <policy-dir>/.env then ./.env if unset
  --init-requirements       Copy requirements.submission.template -> requirements.txt if requirements.txt is missing
  --push                    After build: docker login + tag + push to HF Spaces registry
  --hf-user USER            Hugging Face username (for registry path)
  --hf-space SPACE          HF Space repo name (Docker SDK space)
  --hf-remote-tag TAG       Tag on registry (default: v1)
  --hf-image-ref REF        Full HF image ref (recommended). Example:
                            registry.hf.space/<org>-<space>:latest
                            Use the Space UI “Run with Docker” for the exact value.
  --hf-registry HOST        Default: registry.hf.space

Environment (typical .env on VM — export before run or use --env-file):
  WANDB_API_KEY             For W&B artifact download (never baked into the image)
  HF_TOKEN or HUGGING_FACE_HUB_TOKEN   For docker login when using --push

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-dir) POLICY_DIR="$(cd "$2" && pwd)"; shift 2 ;;
    --dockerfile) DOCKERFILE="$2"; shift 2 ;;
    --wb-project) WB_PROJECT="$2"; shift 2 ;;
    --wb-artifact) WB_ARTIFACT="$2"; shift 2 ;;
    --pretrained-subdir) PRETRAINED_DIR_NAME="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD="1"; shift ;;
    --image-tag) IMAGE_TAG="$2"; shift 2 ;;
    --no-cache) DOCKER_BUILD_EXTRA+=(--no-cache); shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --init-requirements) INIT_REQUIREMENTS="1"; shift ;;
    --push) PUSH="1"; shift ;;
    --hf-user) HF_USER="$2"; shift 2 ;;
    --hf-space) HF_SPACE="$2"; shift 2 ;;
    --hf-remote-tag) HF_REMOTE_TAG="$2"; shift 2 ;;
    --hf-image-ref) HF_IMAGE_REF="$2"; shift 2 ;;
    --hf-registry) HF_REGISTRY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${ENV_FILE}" ]]; then
  load_env_file "${ENV_FILE}"
else
  [[ -f "${POLICY_DIR}/.env" ]] && load_env_file "${POLICY_DIR}/.env"
  [[ -f "./.env" ]] && load_env_file "./.env"
fi

cd "${POLICY_DIR}"

PRETRAINED_PATH="${POLICY_DIR}/${PRETRAINED_DIR_NAME}"

# Plain docker build expects requirements.txt in the context; fill from template if missing.
if [[ ! -f requirements.txt ]]; then
  if [[ ! -f requirements.submission.template ]]; then
    echo "Missing requirements.submission.template in ${POLICY_DIR}" >&2
    exit 1
  fi
  cp requirements.submission.template requirements.txt
  echo "Created requirements.txt from requirements.submission.template"
fi

# Stage BYOP package when not in the build context (workspace: lehome_workspace/lerobot_policy_dino).
LERO_SOURCE="$(cd "${POLICY_DIR}/../.." && pwd)/lerobot_policy_dino"
if [[ ! -d "${POLICY_DIR}/lerobot_policy_dino" ]] || [[ -z "$(ls -A "${POLICY_DIR}/lerobot_policy_dino" 2>/dev/null || true)" ]]; then
  if [[ ! -d "${LERO_SOURCE}" ]]; then
    echo "Missing lerobot_policy_dino under ${POLICY_DIR} and no source at ${LERO_SOURCE}" >&2
    echo "Expected: lehome_workspace/lerobot_policy_dino relative to this challenge checkout." >&2
    exit 1
  fi
  echo "Staging lerobot_policy_dino from ${LERO_SOURCE}..."
  mkdir -p "${POLICY_DIR}/lerobot_policy_dino"
  rsync -a "${LERO_SOURCE}/" "${POLICY_DIR}/lerobot_policy_dino/"
fi

if [[ "${INIT_REQUIREMENTS}" == "1" ]]; then
  if [[ ! -f requirements.txt ]]; then
    if [[ ! -f requirements.submission.template ]]; then
      echo "Missing requirements.submission.template in ${POLICY_DIR}" >&2
      exit 1
    fi
    cp requirements.submission.template requirements.txt
    echo "Created requirements.txt from template — edit it to match your training stack before building."
  else
    echo "requirements.txt already exists; not overwriting."
  fi
fi

if grep -Eq '^[[:space:]]*(torch|torchvision|torchaudio)([[:space:]<>=!~;]|$)' requirements.txt; then
  echo "Do not pin torch/torchvision/torchaudio in requirements.txt for the Blackwell image." >&2
  echo "Dockerfile.submission installs the lerobot-compatible cu128 wheel set after lerobot_policy_dino." >&2
  exit 1
fi

for need in "${DOCKERFILE}" server.py policy.py download_wandb_model.py; do
  if [[ ! -f "${need}" ]]; then
    echo "Missing required file in ${POLICY_DIR}: ${need}" >&2
    echo "Copy server.py from upstream lehome-challenge dummy_docker_policy if needed:" >&2
    echo "  curl -fsSL https://raw.githubusercontent.com/lehome-official/lehome-challenge/main/dummy_docker_policy/server.py -o server.py" >&2
    exit 1
  fi
done

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  if [[ -z "${WB_ARTIFACT}" ]]; then
    echo "--wb-artifact is required unless you pass --skip-download" >&2
    exit 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found; cannot download W&B artifact." >&2
    exit 1
  fi
  echo "Downloading W&B artifact into ${PRETRAINED_PATH} ..."
  python3 download_wandb_model.py --project "${WB_PROJECT}" --artifact "${WB_ARTIFACT}" --out "${PRETRAINED_PATH}"
else
  echo "Skipping W&B download (--skip-download)."
  if [[ ! -d "${PRETRAINED_PATH}" ]] || [[ -z "$(ls -A "${PRETRAINED_PATH}" 2>/dev/null || true)" ]]; then
    echo "Folder missing or empty: ${PRETRAINED_PATH}" >&2
    exit 1
  fi
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker CLI not found." >&2
  exit 1
fi

echo "Building image ${IMAGE_TAG} with -f ${DOCKERFILE} ..."
docker build "${DOCKER_BUILD_EXTRA[@]}" -t "${IMAGE_TAG}" -f "${DOCKERFILE}" "${POLICY_DIR}"

REMOTE_REF=""
if [[ "${PUSH}" == "1" ]]; then
  if [[ -n "${HF_IMAGE_REF}" ]]; then
    REMOTE_REF="${HF_IMAGE_REF}"
  else
    if [[ -z "${HF_USER}" || -z "${HF_SPACE}" ]]; then
      echo "--push requires either --hf-image-ref OR (--hf-user and --hf-space)" >&2
      echo "Tip: HF Spaces image refs are often registry.hf.space/<org>-<space>:latest; the exact repo name may include truncation/suffixes for long space names." >&2
      exit 1
    fi
    # Backwards-compatible fallback. NOTE: HF Spaces repo naming is commonly <org>-<space>,
    # and tags are often :latest / :<commit-sha>. Prefer --hf-image-ref from the Space UI.
    REMOTE_REF="${HF_REGISTRY}/${HF_USER}/${HF_SPACE}:${HF_REMOTE_TAG}"
  fi
  HF_PASS="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
  if [[ -z "${HF_PASS}" ]]; then
    echo "For --push, set HF_TOKEN or HUGGING_FACE_HUB_TOKEN (e.g. in .env)." >&2
    exit 1
  fi
  echo "Logging in to ${HF_REGISTRY} as ${HF_USER} ..."
  echo "${HF_PASS}" | docker login "${HF_REGISTRY}" -u "${HF_USER}" --password-stdin
  echo "Tagging ${IMAGE_TAG} -> ${REMOTE_REF}"
  docker tag "${IMAGE_TAG}" "${REMOTE_REF}"
  echo "Pushing ${REMOTE_REF} ..."
  docker push "${REMOTE_REF}"
  echo "Done. Give organizers: image ${REMOTE_REF} and a read-only HF token scoped to this Space."
else
  echo "Build finished: ${IMAGE_TAG}"
  echo "Test (from lehome-challenge repo root):"
  echo "  docker run --rm -p 8080:8080 --gpus all ${IMAGE_TAG}   # or without --gpus if no GPU"
  echo "  python -m scripts.eval --policy_type docker --docker_url http://localhost:8080 --garment_type top_long --headless --device cpu --enable_cameras"
  if [[ -n "${HF_IMAGE_REF}" ]]; then
    echo "To push: re-run with --push --hf-image-ref ${HF_IMAGE_REF}"
  elif [[ -n "${HF_USER}" && -n "${HF_SPACE}" ]]; then
    echo "To push: re-run with --push --hf-user ${HF_USER} --hf-space ${HF_SPACE} --hf-remote-tag ${HF_REMOTE_TAG}"
  fi
fi
