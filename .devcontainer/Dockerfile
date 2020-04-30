# Start from stable-baselines container
FROM tensorflow/tensorflow:1.14.0-py3

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Add a non-root user with sudo access.
ARG USERNAME=default
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Configure apt
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Install apt deps
    && apt-get -y install \
    sudo \
    git \
    iproute2 \
    procps \
    lsb-release \
    cmake \
    libopenmpi-dev \
    python3-dev \
    zlib1g-dev \
    libsm6 \
    libxrender1 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    #
    # Install pip deps
    && pip --disable-pip-version-check --no-cache-dir install \
    pylint \
    autopep8 \
    stable-baselines[mpi] \
    scikit-learn \
    #
    # Create a non-root user to use if preferred
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Copy code in the container
COPY ./ /home/$USERNAME/drl-baselines-sandbox/
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME/drl-baselines-sandbox/

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Set working directory
WORKDIR /home/$USERNAME/drl-baselines-sandbox/