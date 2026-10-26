FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    MODEL_PATH=/workspace/models/CAD-Coder \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git git-lfs curl wget bzip2 libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 libegl1 build-essential ninja-build util-linux \
    && rm -rf /var/lib/apt/lists/*

# Isolate legacy project's Python dependencies from Vast's changing templates.
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH
RUN conda config --set always_yes yes && conda create -n llava python=3.10 pip && conda create -n cad python=3.10 pip

ARG CAD_CODER_REF=main
RUN git clone https://github.com/anniedoris/cad-coder.git /opt/cad-coder \
    && cd /opt/cad-coder && git checkout "$CAD_CODER_REF" \
    && python3 - <<'PYEDIT'
from pathlib import Path
p = Path("/opt/cad-coder/pyproject.toml")
s = p.read_text()
assert "torch==2.1.2" in s and "torchvision==0.16.2" in s, "Upstream pins changed; review manually"
s = s.replace("torch==2.1.2", "torch==2.7.0").replace("torchvision==0.16.2", "torchvision==0.22.0")
p.write_text(s)
PYEDIT
WORKDIR /opt/cad-coder
# Install compatibility pins before CAD-Coder so pip resolves one coherent set.
# Separate build steps make GitHub Actions show which operation failed.
COPY constraints-llava.txt /opt/app/constraints-llava.txt
ENV PIP_CONSTRAINT=/opt/app/constraints-llava.txt
RUN /opt/conda/envs/llava/bin/python -m pip install --upgrade 'pip<25' 'setuptools<81' wheel \
    && /opt/conda/envs/llava/bin/python -m pip install 'setuptools-scm==8.1.0' \
    && /opt/conda/envs/llava/bin/python -c "import setuptools_scm; print('Legacy setuptools-scm build tooling OK')"
RUN /opt/conda/envs/llava/bin/python -m pip install \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
RUN /opt/conda/envs/llava/bin/python -m pip install \
    'numpy<2' 'huggingface_hub==0.25.2' 'Pillow<11' \
    'peft==0.10.0' 'datasets<3' 'bitsandbytes==0.46.0' tensorboard
RUN /opt/conda/envs/llava/bin/python -m pip install -e .
RUN /opt/conda/envs/llava/bin/python -m pip check \
    && /opt/conda/envs/llava/bin/python -c "import torch,bitsandbytes; print(torch.__version__,torch.version.cuda,bitsandbytes.__version__)"
# CadQuery kept outside inference env, per upstream recommendation.
RUN conda install -n cad -c conda-forge cadquery -y \
    && /opt/conda/envs/cad/bin/pip install trimesh plyfile pandas tqdm \
    && conda clean --all -y

RUN groupadd --system cadexec && useradd --system --gid cadexec --no-create-home cadexec
COPY cad_converter.py /opt/app/cad_converter.py
COPY web_gui.py /opt/app/web_gui.py
COPY start.sh /opt/app/start.sh
RUN chmod +x /opt/app/start.sh && mkdir -p /workspace/models /workspace/outputs
# Do not EXPOSE 7860: use SSH port forwarding only.
CMD ["/opt/app/start.sh"]
