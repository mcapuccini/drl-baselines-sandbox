# Start from stable-baselines container
FROM stablebaselines/stable-baselines-cpu:v2.10.0

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Configure and install deps
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Install apt deps
    && apt-get -y install git iproute2 procps lsb-release \
    #
    # Install pip deps
    && pip --disable-pip-version-check --no-cache-dir install pylint \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog