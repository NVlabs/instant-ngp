#!/bin/bash

## -------------------
## Configuration
## -------------------

# CUDA sub-packages to install.
declare -a CUDA_PACKAGES_IN=(
    "cuda-command-line-tools"
    "cuda-libraries-dev"
    "cuda-nvcc"
)

## -------------------
## Helper Functions
## -------------------

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

## -------------------
## Input Validation and Environment Setup
## -------------------

# Get CUDA version from environment variable 'cuda'
CUDA_VERSION_MAJOR_MINOR="${cuda}"

# Validate CUDA version
if [ -z "$CUDA_VERSION_MAJOR_MINOR" ]; then
  echo "Error: CUDA version not specified. Please set the 'cuda' environment variable (e.g., cuda=12.2)."
  exit 1
fi

CUDA_MAJOR=$(echo "$CUDA_VERSION_MAJOR_MINOR" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION_MAJOR_MINOR" | cut -d. -f2)

# Check for root/sudo
if ! command_exists sudo && [ "$EUID" -ne 0 ]; then
  echo "Error: This script requires root privileges. Please run with sudo."
  exit 1
fi

SUDO_CMD=""
if [ "$EUID" -ne 0 ]; then
  SUDO_CMD="sudo"
fi

# Get Ubuntu version
UBUNTU_VERSION=$(lsb_release -sr)

# Validate Ubuntu version
if [ -z "$UBUNTU_VERSION" ]; then
  echo "Error: Could not determine Ubuntu version."
  exit 1
fi

# Format Ubuntu version for URLs (e.g., 20.04 -> 2004)
UBUNTU_VERSION_FORMATTED=$(echo "$UBUNTU_VERSION" | tr -d '.')

echo "CUDA Version: $CUDA_VERSION_MAJOR_MINOR"
echo "Ubuntu Version: $UBUNTU_VERSION"
echo "Ubuntu Version Formatted: $UBUNTU_VERSION_FORMATTED"

## -------------------
## Install CUDA
## -------------------

# Download and install the CUDA keyring
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION_FORMATTED}/x86_64/cuda-keyring_1.1-1_all.deb"
echo "Downloading CUDA keyring from: $KEYRING_URL"
wget -nv "$KEYRING_URL" -O cuda-keyring.deb
$SUDO_CMD dpkg -i cuda-keyring.deb
rm cuda-keyring.deb

# Update package list
echo "Updating package list..."
$SUDO_CMD apt-get update

# Construct the list of CUDA packages to install
CUDA_PACKAGES=""
for package in "${CUDA_PACKAGES_IN[@]}"; do
  CUDA_PACKAGES+=" ${package}-${CUDA_MAJOR}-${CUDA_MINOR}"
done

# Special handling for nvcc in older versions
if (( $(echo "$CUDA_MAJOR < 9" | bc -l) )); then
  CUDA_PACKAGES+=" cuda-nvcc-${CUDA_MAJOR}-${CUDA_MINOR}"
fi

# Install CUDA packages
echo "Installing CUDA packages: $CUDA_PACKAGES"
$SUDO_CMD apt-get install -y --no-install-recommends $CUDA_PACKAGES

if [ $? -ne 0 ]; then
  echo "Error: Failed to install CUDA packages."
  exit 1
fi

## -------------------
## Environment Variables
## -------------------

CUDA_PATH="/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}"
echo "CUDA_PATH=$CUDA_PATH"

# Update environment variables for the current shell
export CUDA_PATH
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

# Verify installation (optional)
echo "Verifying installation..."
if command_exists nvcc; then
  nvcc --version
else
  echo "Warning: nvcc not found. Installation might be incomplete."
fi

# Update environment variables for GitHub Actions (if applicable)
if [ -n "$GITHUB_ACTIONS" ]; then
  echo "Setting environment variables for GitHub Actions..."
  echo "CUDA_PATH=$CUDA_PATH" >> "$GITHUB_ENV"
  echo "$CUDA_PATH/bin" >> "$GITHUB_PATH"
  echo "LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH" >> "$GITHUB_ENV"
fi

echo "CUDA installation complete."
