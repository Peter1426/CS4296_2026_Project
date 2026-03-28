#!/bin/bash

# Exit on error
set -e  

# Check for GPU flag (default false)
USE_GPU=${1:-false}

echo "=========================================="
echo "Image Retrieval Benchmark on AWS EC2"
echo "=========================================="

# Install dependencies
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip git wget unzip python3-venv

# Install NVIDIA drivers if in GPU mode
if [ "$USE_GPU" = "true" ]; then
    echo "Installing NVIDIA drivers for GPU support..."
    sudo apt-get install -y nvidia-driver-535 nvidia-utils-535
    sudo apt-get install -y nvidia-cuda-toolkit
fi

# Clone or update repository (if have not done so)
if [ ! -d "CS4296_2026_Project" ]; then
    echo "Cloning repository..."
    git clone https://github.com/Peter1426/CS4296_2026_Project.git
fi

cd CS4296_2026_Project

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install Python packages inside virtual environment
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Download Flickr8k dataset if not exist
if [ ! -d "data/images" ] || [ -z "$(ls -A data/images 2>/dev/null)" ]; then
    echo "Downloading Flickr8k dataset..."
    mkdir -p data/images
    mkdir -p data/queries
    
    # Download from the GitHub release
    echo "Downloading flickr8k.zip (this may take a few minutes)..."
    wget --progress=bar:force:noscroll -O flickr8k.zip \
        "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"
    
    echo "Extracting images..."
    unzip -q flickr8k.zip -d data/
    
    # The zip contains an 'Images' folder
    if [ -d "data/Images" ]; then
        mv data/Images/* data/images/
        rmdir data/Images
    fi
    
    # Clean up
    rm -f flickr8k.zip
    
    # Check if there is any images
    IMAGE_COUNT=$(ls data/images | wc -l)
    if [ "$IMAGE_COUNT" -eq 0 ]; then
        echo "ERROR: Failed to extract images. Please check the download."
        exit 1
    fi

    echo "Splitting dataset into database and query sets..."
    
    cd data/images
    
    # Create list of all images
    ls *.jpg > all_images.txt
    TOTAL_IMAGES=$(wc -l < all_images.txt)
    
    # Use 100 images as queries
    QUERY_COUNT=100
    DATABASE_COUNT=$((TOTAL_IMAGES - QUERY_COUNT))
    
    echo "  Total images: $TOTAL_IMAGES"
    echo "  Database images: $DATABASE_COUNT"
    echo "  Query images: $QUERY_COUNT"
    
    # Move the LAST 100 images to queries folder (different from database)
    tail -n $QUERY_COUNT all_images.txt | while read img; do
        mv "$img" ../queries/
    done
    
    # Clean up
    rm all_images.txt
    cd ../..
    
    echo "Dataset ready:"
    echo "  Database images: $(ls data/images | wc -l) images"
    echo "  Query images: $(ls data/queries | wc -l) images"
else
    echo "Dataset already exists. Skipping download."
    echo "  Database images: $(ls data/images | wc -l)"
    echo "  Query images: $(ls data/queries | wc -l)"
fi

# Create results directory if it doesn't exist
mkdir -p results

# Run benchmark
if [ "$USE_GPU" = "true" ]; then
    echo "Running with GPU mode..."
    python3 src/benchmark.py \
        --dataset ./data/images \
        --queries ./data/queries \
        --output ./results/benchmark_results.json \
        --gpu
else
    echo "Running CPU benchmark..."
    python3 src/benchmark.py \
        --dataset ./data/images \
        --queries ./data/queries \
        --output ./results/benchmark_results.json
fi

echo "Benchmark complete! Results in ./results/"

# Deactivate virtual environment
deactivate
