# syntax=docker/dockerfile:1.3

# Check supported tags for CUDA and UBUNTU versions:
# https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
ARG CUDA_VERSION=11.3.1
ARG UBUNTU_VERSION=20.04
ARG APP_ENV="dev"

FROM nvidia/cudagl:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as install

# Fix Cuda Path
ENV CUDA_ROOT /usr/local/cuda
ENV PATH $PATH:$CUDA_ROOT/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$CUDA_ROOT/lib64:$CUDA_ROOT/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib
ENV LIBRARY_PATH /usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:/usr/local/cuda/lib$LIBRARY_PATH

RUN rm -f /etc/apt/sources.list.d/cuda.list
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

ENV DEPENDENCIES="/dependencies"
WORKDIR ${DEPENDENCIES}

# basic packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -y update                && \
    apt-get -y upgrade               && \
    apt-get install -y                  \
    software-properties-common      \
    sudo                            \
    build-essential                 \
    wget                            \
    curl                            \
    git                             \
    swig

#-------------------------------------------------------------
#       C++
#-------------------------------------------------------------

# CMAKE
RUN apt purge -y --auto-remove cmake && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && \
    apt-get install -y cmake

# C++ 20
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt-get install -y gcc-9 g++-9 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9 && \
    update-alternatives --config gcc

#-------------------------------------------------------------
#       Python Installation
#-------------------------------------------------------------
ARG PYTHON_VERSION=3.10
RUN add-apt-repository ppa:deadsnakes/ppa       && \
    apt-get -y update                           && \
    apt-get install -y                             \
    python${PYTHON_VERSION}                        \
    python${PYTHON_VERSION}-dev                    \
    libpython${PYTHON_VERSION}-dev                 \
    python${PYTHON_VERSION}-tk                     \
    python${PYTHON_VERSION}-distutils              \
    python3-setuptools                             \
    python3-wheel

#----------------------------------------------
# Pip binding
#----------------------------------------------

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python${PYTHON_VERSION} get-pip.py

# Force python${PYTHON_VERSION} instead python3.6
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

#----------------------------------------------
#   COLMAP Installation
#----------------------------------------------

WORKDIR ${DEPENDENCIES}

RUN echo "Installing apt packages..." \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt -y update --no-install-recommends \
    && apt -y install --no-install-recommends \
    git \
    wget \
    ffmpeg \
    tk-dev \
    libxi-dev \
    libc6-dev \
    libbz2-dev \
    libffi-dev \
    libomp-dev \
    libssl-dev \
    zlib1g-dev \
    libcgal-dev \
    libgdbm-dev \
    libglew-dev \
    qtbase5-dev \
    checkinstall \
    libglfw3-dev \
    libeigen3-dev \
    libgflags-dev \
    libxrandr-dev \
    libopenexr-dev \
    libsqlite3-dev \
    libxcursor-dev \
    libcgal-qt5-dev \
    libxinerama-dev \
    libboost-all-dev \
    libfreeimage-dev \
    libncursesw5-dev \
    libatlas-base-dev \
    libqt5opengl5-dev \
    libgoogle-glog-dev \
    libsuitesparse-dev \
    libreadline-gplv2-dev \
    && apt autoremove -y \
    && apt clean -y \
    && export DEBIAN_FRONTEND=dialog

# ARG CERES_SOLVER_VERSION=2.0.0
# RUN echo "Installing Ceres Solver ver. ${CERES_SOLVER_VERSION}..." \
#     && git clone https://github.com/ceres-solver/ceres-solver \
#     && cd ./ceres-solver \
#     && git checkout ${CERES_SOLVER_VERSION} \
#     && mkdir ./build \
#     && cd ./build \
#     && cmake ../ -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
#     && make -j2 \
#     && make install

# ARG COLMAP_VERSION=3.7
# RUN echo "Installing COLMAP ver. ${COLMAP_VERSION}..." \
#     && git clone https://github.com/colmap/colmap \
#     && cd ./colmap \
#     && git checkout ${COLMAP_VERSION} \
#     && mkdir ./build \
#     && cd ./build \
#     && cmake ../ \
#     && make -j2 \
#     && make install \
#     && colmap -h

#----------------------------------------------
#   NVIDIA-Optix
#----------------------------------------------

WORKDIR ${DEPENDENCIES}

ENV OptiX_INSTALL_DIR=${DEPENDENCIES}/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64
COPY .devcontainer/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64.sh NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64.sh
RUN mkdir -p ${OptiX_INSTALL_DIR} && \
    bash NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64.sh --skip-license --prefix=${OptiX_INSTALL_DIR}

#----------------------------------------------
#   Dependencies packages
#----------------------------------------------

RUN apt-get -y update &&       \
    apt-get -y install         \
    libopenexr-dev             \
    libxi-dev                  \
    libglfw3-dev               \
    libglew-dev                \
    libomp-dev                 \
    libxinerama-dev            \
    libxcursor-dev             \
    zlib1g-dev

#-------------------------------------------------------------
#       Instant-NGP Installation
#-------------------------------------------------------------

ENV INSTANT_NGP_3DML_WORKDIR="/workspaces/3dml-instant-ngp"
WORKDIR ${INSTANT_NGP_3DML_WORKDIR}

COPY .devcontainer/requirements.txt .devcontainer/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r .devcontainer/requirements.txt


#----------------------------------------------
# Install Utils3DML
#----------------------------------------------

# Get Git Auth
ARG GIT_ACCESS_TOKEN
ENV GIT_ACCESS_TOKEN=$GIT_ACCESS_TOKEN
WORKDIR "${DEPENDENCIES}/Utils3DML"

# 3dml-utils Installation
COPY .devcontainer/.env .env
SHELL ["/bin/bash", "-c"]
RUN export UTILS_3DML_VERSION=$(echo "$(cut -d'=' -f2 <<< $(grep UTILS_3DML_VERSION .env))") && \
    git clone https://${GIT_ACCESS_TOKEN}@github.com/checkandvisit/3dml-utils.git && \
    cd 3dml-utils && git checkout ${UTILS_3DML_VERSION}
RUN --mount=type=cache,target=/root/.cache/pip \
    cd 3dml-utils && pip3 install -e .


#-------------------------------------------------------------
#       3D_NGP Installation
#-------------------------------------------------------------

ENV PYTHONPATH $PYTHONPATH:$INSTANT_NGP_3DML_WORKDIR

WORKDIR $INSTANT_NGP_3DML_WORKDIR
COPY . .

#-------------------------------------------------------------------------------------
#                       DEV INSTALLATION
#-------------------------------------------------------------------------------------

FROM install as dev-instant-ngp

#----------------------------------------------
#   Vulkan
#----------------------------------------------

RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -       && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-focal.list                                   \
    http://packages.lunarg.com/vulkan/lunarg-vulkan-focal.list                                  && \
    apt-get -y update                                                                           && \
    apt-get -y install vulkan-sdk

#----------------------------------------------
#   Dev utils
#----------------------------------------------

RUN apt-get -y update &&       \
    apt-get -y install         \
    awscli                     \
    nano                       \
    ffmpeg                     \
    unzip

COPY .devcontainer/requirements_dev.txt .devcontainer/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r .devcontainer/requirements_dev.txt

#-------------------------------------------------------------------------------------
#                       BUILD INSTALLATION
#-------------------------------------------------------------------------------------

FROM install as build-instant-ngp

COPY . .

ARG CPU_SIZE=4
ENV TCNN_CUDA_ARCHITECTURES=75
RUN cmake . -B build && \
    cmake --build build --config RelWithDebInfo -j${CPU_SIZE}


#-------------------------------------------------------------------------------------
#                       Finalisation
#-------------------------------------------------------------------------------------

FROM ${APP_ENV}-instant-ngp as final

ENV PYTHONPATH=$PYTHONPATH:"/workspaces/3dml-instant-ngp/build/"

## Switch to non-root user
ENV USER=dock
ENV GROUP=sudo

RUN useradd -ms /bin/bash ${USER} && \
    usermod -aG ${GROUP} ${USER}

# Cleaning Part
USER root
RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/*

# Resolve authorization problem
RUN echo "${USER} ALL=(ALL) NOPASSWD: ALL" \
    >/etc/sudoers.d/${USER} && \
    chmod 0440 /etc/sudoers.d/${USER}

# Final User
USER dock
