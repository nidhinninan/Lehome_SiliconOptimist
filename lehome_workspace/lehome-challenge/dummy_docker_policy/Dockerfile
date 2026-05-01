# Minimal CPU-only image for quick local smoke tests.
# GPU / Blackwell (sm_120): use Dockerfile.submission + build_docker_submission.sh (CUDA 12.8 PyTorch base).
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY server.py policy.py ./
EXPOSE 8080
CMD ["python", "policy.py"]
