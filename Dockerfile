FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/scrfd-fullsearch-kaggle

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/scrfd-fullsearch-kaggle

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install "numpy<2" cython && \
    python -m pip install \
      -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html \
      mmcv-full==1.2.7 && \
    python -m pip install -r requirements.txt && \
    python -m pip install -e .

CMD ["/bin/bash"]

