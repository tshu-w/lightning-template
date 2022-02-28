FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG CONDA_ENV=env

SHELL ["/bin/bash", "-c"]

RUN apt update --fix-missing -qq && \
  apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    curl \
    wget && \
  apt autoremove -y && \
  apt-get clean && \
  rm -rf /root/.cache && \
  rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:/opt/conda/bin:${PATH}
RUN curl -o miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
  /bin/bash miniconda3.sh -b -p /opt/conda && \
  rm miniconda3.sh && \
  conda update -n base -c defaults conda && \
  conda create -n ${CONDA_ENV} && \
  conda clean -ay && \
  echo "source activate ${CONDA_ENV}" >> ~/.bashrc

COPY environment.yaml requirements.txt ./
RUN \
  ([ ! -f environment.yaml ] || conda env update -n ${CONDA_ENV} -f environment.yaml) && \
  ([ ! -f requirements.txt ] || pip install --no-cache-dir -r requirements.txt) && \
  conda clean -ay && \
  rm requirements.txt environment.yaml
