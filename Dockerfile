# TopoDIM — GPU dev/training + coding agents (Claude Code, Codex, OpenCode)
#
# Build (uses local tag nvidia/cuda:12.8.0-devel-ubuntu22.04 by default):
#   docker build -t topodim:dev .
# Campus mirror:
#   docker build --build-arg BASE_IMAGE=ngc.nju.edu.cn/nvidia/cuda:12.8.0-devel-ubuntu22.04 -t topodim:dev .
#
# Run (GPU, mount repo for live editing):
#   docker run --gpus all -it --rm \
#     -v "$(pwd):/workspace" \
#     -v topodim-pip-cache:/root/.cache/pip \
#     -v topodim-hf-cache:/root/.cache/huggingface \
#     --add-host=host.docker.internal:host-gateway \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     -e ANTHROPIC_API_KEY \
#     -e OPENAI_API_KEY \
#     topodim:dev
#
# Inside the container, run experiments from /workspace, e.g.:
#   python -u experiments/run_mmlu_pro.py --llm_name <model> ...

# Must match an image you already have locally (no cudnn8 suffix for 12.8).
ARG BASE_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG NODE_MAJOR=22

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    PATH="/root/.local/bin:/root/.claude/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg git git-lfs openssh-client \
    build-essential cmake pkg-config \
    libssl-dev libffi-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev liblzma-dev \
    libgl1-mesa-glx libglib2.0-0 \
    vim less ripgrep \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv python3.10-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN python3.10 -m pip install --upgrade pip setuptools wheel \
    && grep -vE '^(pywin32|pywin32-ctypes|pywinpty|win32-setctime|tensorflow|tensorflow-estimator|tensorflow-io-gcs-filesystem|# torch|^torch==|^torchaudio|^torchvision|^torch-geometric|^torch_cluster|^torch_scatter|^torch_sparse|^pyg-lib)' \
        /tmp/requirements.txt > /tmp/requirements-docker.txt \
    && python3.10 -m pip install \
        torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && python3.10 -m pip install \
        pyg-lib==0.4.0+pt23cu121 \
        torch-scatter==2.1.2+pt23cu121 \
        torch-sparse==0.6.18+pt23cu121 \
        torch-cluster==1.6.3+pt23cu121 \
        torch-geometric==2.6.1 \
        -f https://data.pyg.org/whl/torch-2.3.0+cu121.html \
    && python3.10 -m pip install -r /tmp/requirements-docker.txt \
    && python3.10 -m pip install ollama \
    && rm -f /tmp/requirements.txt /tmp/requirements-docker.txt

RUN npm install -g \
        @anthropic-ai/claude-code@latest \
        @openai/codex@latest \
        opencode-ai@latest \
    && curl -fsSL https://claude.ai/install.sh | bash || true \
    && curl -fsSL https://opencode.ai/install | bash || true

COPY . /workspace

RUN python -c "import torch; import torch_geometric; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    && (claude --version || true) \
    && (codex --version || true) \
    && (opencode --version || opencode version || true)

CMD ["/bin/bash"]
