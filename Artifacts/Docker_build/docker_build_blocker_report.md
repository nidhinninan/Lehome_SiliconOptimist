# Docker Build Blocker Report
**Date**: 2026-04-30  
**VM Hostname**: d44eecb8e428 (Cursor/LeHome challenge VM)

---

## Root Cause (1 sentence)

This VM is itself running **inside a Docker container** (confirmed by `/.dockerenv` and cgroup paths), and the container was **not launched with `--privileged`**, so the Linux kernel **blocks `unshare(CLONE_NEWUSER)` / `CLONE_NEWNS`** — the exact system calls that both Docker's build engine and Buildah require to create isolated filesystem layers for each `RUN` instruction in a Dockerfile.

---

## Evidence

| Check | Result |
|---|---|
| `/.dockerenv` present | YES — we are inside a Docker container |
| cgroup path | `/docker/d44eecb8e428...` |
| `CAP_SYS_ADMIN` | **MISSING** (hex decode of CapEff confirms) |
| `unshare --user echo ok` | `Operation not permitted` |
| `unshare --mount echo ok` | `Operation not permitted` |
| Docker daemon start | Starts OK with `--iptables=false --bridge=none` |
| `docker build` (BuildKit) | Fails: `mount ... operation not permitted` |
| `docker build` (legacy `DOCKER_BUILDKIT=0`) | Fails: `unshare: operation not permitted` |
| `buildah build --isolation chroot` | Fails: `Error during unshare(CLONE_NEWUSER): Operation not permitted` |
| `kernel.unprivileged_userns_clone` | = 1 (set), but seccomp/apparmor profile **still blocks it at the syscall level** |

---

## What Would Fix It

### Option A — Switch to a proper non-containerized VM (RECOMMENDED)
Any standard Ubuntu/Debian VM where Docker was installed via `apt` and `systemctl start docker` works.  
That is: **get a machine where `docker info` returns cleanly without errors and `docker run hello-world` works.**

Examples:
- A new DigitalOcean/AWS/GCP/Azure Ubuntu 22.04 droplet/instance
- A local Linux machine with Docker Desktop or Docker Engine installed
- GitHub Actions / any CI runner (GitHub, GitLab, CircleCI all work out of the box)

### Option B — Re-launch this container with `--privileged`
If whoever controls this Cursor VM can restart it with `--privileged` (or add `SYS_ADMIN` + relax the seccomp profile), Docker builds would work.

### Option C — GitHub Actions (zero local setup needed)
Push the `dummy_docker_policy` directory to a private GitHub repo and add a workflow that builds and pushes to `nninspaceexp/lehome_silicon-optimists:FullDP-v1` automatically. Free and fast.

---

## Recommended Next Step

**Use GitHub Actions** — it is the fastest path requiring no new VM provisioning.  
A minimal workflow is all that is needed (see below).

```yaml
# .github/workflows/build-push.yml
name: Build & Push Submission Image
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: lehome-challenge/dummy_docker_policy
          file: lehome-challenge/dummy_docker_policy/Dockerfile.submission
          push: true
          tags: nninspaceexp/lehome_silicon-optimists:FullDP-v1
```

**Required GitHub secrets to add:**
- `DOCKERHUB_USERNAME` = `nninspaceexp`
- `DOCKERHUB_TOKEN` = your Docker Hub Access Token (read/write)

---

## Summary Decision

| Option | Effort | Speed | Recommended? |
|---|---|---|---|
| Switch to a real Ubuntu VM | Medium (provision VM) | ~30 min | Yes |
| GitHub Actions workflow | Low (push repo + secrets) | ~15 min | **Yes (fastest)** |
| Re-launch container with `--privileged` | Depends on who manages infra | Unknown | If available |
| Fix in current environment | **Not possible** | N/A | No |
