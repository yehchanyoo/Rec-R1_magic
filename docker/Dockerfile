FROM nvcr.io/nvidia/rapidsai/notebooks:25.04-cuda12.8-py3.12

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

USER root

# Install git and other system packages
RUN apt-get update && \
    apt-get install -y git default-jdk default-jre vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER rapids
# Install uv
RUN pip3 install --no-cache-dir uv

# Install critical packages
RUN uv pip install --system --no-cache-dir vllm ray

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
RUN git clone https://github.com/yehchanyoo/Rec-R1_magic.git
WORKDIR /home/rapids/Rec-R1_magic
RUN uv pip install --system --no-build-isolation -e .
RUN git remote set-url origin git@github.com:yehchanyoo/Rec-R1_magic.git
