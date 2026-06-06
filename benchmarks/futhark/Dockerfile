FROM docker.io/nvidia/cuda:13.1.0-devel-ubuntu24.04

SHELL ["/bin/bash", "-c"]

WORKDIR /root

# Add a nice prompt for bash
RUN echo "export PS1='\[\e]0;\u@\h: \w\a\]\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc

# Install basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg apt-transport-https lsb-release wget tar gzip python3 python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# Install nsys
RUN apt-get update && \
    apt-get install -y nsight-systems-2025.6.3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Verify nsys CLI is available
RUN which nsys || nsys --version

WORKDIR /tmp

# Install precompiled Futhark binary
RUN wget "https://futhark-lang.org/releases/futhark-0.26.3-linux-x86_64.tar.xz" \
 && tar -xvf futhark-0.26.3-linux-x86_64.tar.xz \
 && mv /tmp/futhark-0.26.3-linux-x86_64/bin/futhark /usr/local/bin/ \
 && rm -r futhark-0.26.3-linux-x86_64 futhark-0.26.3-linux-x86_64.tar.xz

# Install Anaconda
RUN mkdir -p /root/miniconda3 \
 && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh \
 && bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 \
 && rm /root/miniconda3/miniconda.sh \
 && source /root/miniconda3/bin/activate \
 && conda tos view \
 && conda tos accept

# Needs to be done manually, because running it via the Dockerfile installs a non-CUDA version of PyTorch
# Install ParPy
#RUN git clone https://github.com/parpy-lib/ParPy.git \
# && cd /tmp/ParPy \
# && source /root/miniconda3/bin/activate \
# && conda env create -f benchmarks/cuda-env.yml -y \
# && conda activate cuda-parpy-env \
# && pip install .

# podman build --format docker --rm -t futhark-container .
# podman run --rm -it --volume .:/root/custom --device "nvidia.com/gpu=all" futhark-container bash
