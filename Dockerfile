# UPAR 2027 Track 3 — training & inference environment.
# Base matches the environment used for the final submission (torch 2.13.0, CUDA 13.0,
# Blackwell GPU). PyTorch is provided by the base image; requirements_train.txt adds the rest.
#
# Build:  docker build -t upar-track3 .
# Run:    docker run --gpus all -it --rm upar-track3 bash
#         (mount datasets and outputs as needed, e.g.
#          -v /path/to/annotations:/ann -v /path/to/images:/images -v $PWD/outputs:/workspace/outputs)
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
WORKDIR /workspace

COPY requirements_train.txt .
# The base image's Python is externally managed (PEP 668); installing into it is
# safe inside a container.
RUN pip install --no-cache-dir --break-system-packages -r requirements_train.txt \
 && python -c "import torch, numpy, PIL, yaml, pycocotools; print('env OK, torch', torch.__version__)"

COPY train.py .
COPY pose_src pose_src
COPY configs configs
COPY scripts scripts
COPY submission submission
COPY tests tests
COPY local local

# Self-check with mock data (CPU-only, no dataset or GPU required)
RUN python -m pytest tests/ -q

# Optional: bake the released trained weights into the image — download and rejoin them
# from the GitHub release into ./weights first (see README "Trained weights"), then uncomment:
# COPY weights /workspace/weights

CMD ["bash"]
