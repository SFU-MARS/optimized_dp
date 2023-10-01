ARG UBUNTU_VER=20.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64

FROM jupyter/minimal-notebook:ubuntu-${UBUNTU_VER}

# System packages
USER root
RUN apt-get update && apt-get install -yq curl wget jq vim git

# Use the above args during building https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda install -c plotly plotly=5.13.0
RUN conda install ipywidgets

# Install jupyterlab extensions
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
RUN apt-get install -y nodejs
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly jupyter-matplotlib

WORKDIR /lab

# Setup optimized_dp
COPY . .
RUN conda init
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "odp", "/bin/bash", "-c"]
RUN pip install -e .
RUN conda install ipywidgets
RUN conda install ipykernel
RUN conda install nbformat
RUN ipython kernel install --name=odp

# Ubuntu 20.04-specific dependency
RUN apt-get install -yq libtinfo5

# Not necessary, but needed in most projects
RUN conda install scipy

USER 1001
