## Blackwell GPU build variant

This package updates CUDA to 12.8.1, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128 and bitsandbytes 0.46.0. Because upstream CAD-Coder pins torch 2.1.2, the Dockerfile patches its `pyproject.toml` before `pip install -e .`. Other CadQuery, Gradio, and tunnel functionality is retained from the prior release.

**Not GPU tested.** CAD-Coder legacy LLaVA/Transformers may need additional runtime compatibility changes. Rebuild image on GitHub, then re-create the Vast instance to pull updated `latest`.

### GPU verification
```bash
/opt/conda/envs/llava/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); print(torch.cuda.get_device_name(0))"
tail -n 60 /workspace/cad-coder.log
```

# CAD-Coder on Vast.ai — Gradio + CAD export

This image pins CUDA 12.1.1, Python 3.10, PyTorch 2.1.2, torchvision 0.16.2, Transformers version from the upstream CAD-Coder package, and provides:

- Image-to-CadQuery Python generation with a 4-bit model loader.
- Automatic CadQuery STEP (`.step`) export and STL (`.stl`) 3D browser preview.
- Downloadable Python and STEP files, stored under `/workspace/outputs/<job-id>/`.
- A browser interface on `127.0.0.1:7860`, accessible over a private SSH tunnel.

**Status:** source-level and Python syntax checked, **not** yet built or GPU tested. GPU-driver compatibility, repository dependencies, model inference, conversion reliability and memory usage require testing on your selected Vast.ai host. CAD-Coder is a research model: generated code may be invalid or produce no usable solid.

## Security warning

Generated CadQuery code is **executed** automatically to create the STEP and STL outputs. The converter runs as an unprivileged Linux user (`cadexec`) in a separate subprocess, with CPU/memory/file-size limits and a wall-clock timeout. **This is NOT a secure sandbox:** the subprocess is still inside the same Docker container, and no network namespace or complete filesystem isolation is provided. Only use with inputs you trust, on a private single-user instance. Do not expose Gradio to the public internet or accept code from unknown users. A production deployment needs a separate isolated container/VM with no network, read-only source mounts, and strict filesystem limits.

## Publish image using GitHub

1. Create a GitHub repository (`cad-coder-vast`). Upload all files from this folder, including `.github/workflows/publish.yaml`.
2. GitHub **Actions** > **Build and publish CAD-Coder image** > Run workflow (or push changes to `main`).
3. Make your GitHub Container Registry (GHCR) package public.
4. Set Vast.ai custom Docker image to `ghcr.io/YOUR_USERNAME/cad-coder-vast:latest`.

The registry name is lowercase; Docker image publishing requires a successful GitHub Action, not merely hosting a Dockerfile on GitHub.

## Vast.ai template

- **Image:** `ghcr.io/YOUR_USERNAME/cad-coder-vast:latest`.
- **Launch mode:** Interactive shell server (SSH).
- **On-start script** (Vast SSH mode overrides the Docker CMD):

  ```bash
  #!/bin/bash
  mkdir -p /workspace/outputs
  nohup /opt/app/start.sh > /workspace/cad-coder.log 2>&1 &
  ```

- **GPU:** 24 GB VRAM is a starting target for 4-bit testing, **not a guarantee** for this 13B model. 48 GB is a more conservative target.
- **Disk:** 100 GB or more initially; preserve `/workspace` to avoid repeating model downloads.
- **Additional published ports:** none (private SSH forwarding).

## MobaXterm tunnel on Windows

1. Get the Vast.ai instance's actual **SSH hostname**, **SSH port**, and **SSH username** from the Vast instance page.
2. In MobaXterm, use **Tools > SSH tunnels > New SSH tunnel**, choose **Local port forwarding**.
3. Fill in:

   | Field | Value |
   | --- | --- |
   | Forwarded port | `7860` |
   | Remote server | `127.0.0.1` |
   | Remote port | `7860` |
   | SSH server | Vast-provided SSH hostname |
   | SSH login | Vast-provided user (usually `root`) |
   | SSH port | Vast-provided SSH port |

4. Select your SSH private key if needed, start the tunnel, and open `http://localhost:7860` on Windows.

## Troubleshooting

On the Vast.ai instance:

```bash
tail -n 100 /workspace/cad-coder.log
nvidia-smi
ss -ltn | grep 7860
ls -lah /workspace/outputs
```

If **STEP conversion failed**, inspect the generated Python in the Gradio interface. Code produced by CAD-Coder can contain unsupported imports, incomplete geometry, invalid topology, or use a result variable that the converter cannot find. The converter looks for top-level CadQuery Workplane/Shape objects, preferring `result`, `model`, `part`, `solid`, `shape`, or `obj`, and also supports other top-level names. It may need adaptation to specific output styles.

**Note:** The 3D browser preview is derived from an STL mesh, but the downloaded STEP comes from the CadQuery solid (not a conversion of the STL).

## GitHub Actions build diagnostics

Dependency installation is split into separate Docker layers. If a build fails, expand the failing `RUN` step and locate the FIRST `ERROR:` or `ResolutionImpossible` line in the raw log, rather than the final `buildx failed` summary. This revision is a compatibility fix, not a guaranteed GPU-tested build.


## Legacy wavedrom metadata build fix (October 2026)

The Dockerfile now pins `setuptools-scm==8.1.0` and `setuptools<81` before installing CAD-Coder, with matching pip constraints. This avoids a failing `wavedrom` legacy `setup.py egg_info` path where the newer setuptools-scm 10.3.4 attempts to import `vcs_versioning` from an incompletely prepared build environment. The image has not been built and tested on a GPU here.
