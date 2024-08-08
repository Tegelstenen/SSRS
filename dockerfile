FROM --platform=linux/amd64 nvcr.io/nvidia/tensorflow:22.11-tf2-py3

# Update the package list and install necessary packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    curl \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create and activate a virtual environment
RUN python3 -m venv /env
RUN /bin/bash -c "source /env/bin/activate"

# Copy the requirements file and install dependencies
COPY models/modules models/modules
COPY models/utils models/utils
COPY models/main.py models/main.py
COPY requirements.txt requirements.txt
COPY config.yaml config.yaml
RUN pip install -r requirements.txt

# Download the init-ssh.sh script
RUN curl -o ./init-ssh.sh https://raw.githubusercontent.com/aixia-aiqu/aiqu-utils/main/init-ssh.sh

# Make the init-ssh script executable
RUN chmod +x ./init-ssh.sh

# Remove unnecessary files and directories
RUN rm -rf docker-examples nvidia-examples NVIDIA_Deep_Learning_Container_License.pdf README.md

EXPOSE 22

CMD ["tail", "-f", "/dev/null"]