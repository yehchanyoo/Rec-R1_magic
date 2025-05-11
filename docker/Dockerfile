# Issues with blis package below 3.12
FROM nvcr.io/nvidia/rapidsai/notebooks:25.04-cuda12.8-py3.12

# Install git and other system packages
USER root
RUN apt-get update && \
    apt-get install -y build-essential git default-jdk default-jre vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER rapids
ENV CC=gcc
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Install uv
RUN pip3 install --no-cache-dir uv

# Install critical packages
RUN uv pip install --system --no-cache-dir vllm==0.6.3 ray

# Install common packages
RUN uv pip install --system --no-cache-dir \
        accelerate \
        codetiming \
        datasets \
        dill \
        hydra-core \
        numpy \
        pybind11 \
        tensordict \
        ipython \
        matplotlib \
        seaborn \
        blis

# Final utilities
RUN uv pip install --system wandb py-spy

# Install pyserini
RUN uv pip install --system --no-cache-dir pyserini

# Install faiss and dependencies using conda
RUN conda install -y -c pytorch -c nvidia -c rapidsai -c conda-forge \
        libnvjitlink \
        cuda-version=12.4 \
        libcuvs=25.4.0 \
        faiss-gpu-cuvs=1.11.0 && \
    conda clean -afy

# Install custom Verl package
ENV MAX_JOBS=4
ENV NINJA_FLAGS="-j4"

# Copy from local
COPY --chown=rapids:conda Rec-R1_magic /home/rapids/Rec-R1_magic 
WORKDIR /home/rapids/Rec-R1_magic
RUN uv pip install --system --no-build-isolation -e .
