# CS4296 - Image Retrieval Benchmark on AWS EC2

## Project Overview

This is an individual project from **CS4296 Cloud Computing** at year 2026. This project benchmarks the performance of an image retrieval system over different Amazon EC2 instances.

The image retrieval system consists of two components:
- **Feature extraction**: *MobileNetV2* converts images to 1280-dimensional feature vectors
- **Similarity search**: *FAISS* (Facebook AI Similarity Search) performs efficient vector search with CPU and GPU support

## Key Feature

- Automatic dataset download
    - The project uses the **Flickr8k** dataset (8,091 images, 1 caption)
- Automatic dataset splitting
    - The **Flickr8k** dataset are split into **Database**: 7,991 images and **Queries**: 100 images
- CPU and GPU benchmark modes
- Multi-GPU support (auto-detects)
- Comprehensive performance metrics:
    - Query latency (ms)
    - Query throughput (QPS)
    - Index build time (sec)
    - Memory usage (MB)
- Result saved in JSON an CSV formats

## Evaluate EC2 instances

The system is deployed and benchmarked on six EC2 instance types (t3.micro, t3.medium, t3.large, c5.large, c5.xlarge, r5.large) to evaluate.

| Type | vCPU | RAM | Purpose |
|------|------|-----|---------|
| t3.micro | 2 | 1 GB | CPU baseline (RAM impact test) |
| t3.medium | 2 | 4 GB | CPU baseline (RAM impact test) |
| t3.large | 2 | 8 GB | Burstable | CPU baseline (RAM impact test) |
| c5.large | 2 | 4 GB | CPU compute-optimized (CPU scaling test) |
| c5.xlarge | 4 | 8 GB | Compute-optimized (CPU scaling test) |
| r5.large | 2 | 16 GB | CPU memory-optimized (Memory bandwidth test) |

### GPU Support

GPU acceleration is implemented using FAISS's `index_cpu_to_all_gpus()` function, which automatically detects and utilizes all available GPUs. ***Although testing with GPU instances is not possible due to AWS Academy account limitations, the code is fully ready for GPU deployment.***

## Dataset

This project uses the **Flickr8k dataset** (8,091 images). The benchmark script automatically downloads the dataset from one of the following sources:

1. **GitHub Release** (primary): https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip
2. **Kaggle API** (fallback): Requires Kaggle account and API key
3. **Manual download** (last resort): Instructions provided if automatic download fails

### If Automatic Download Fails

1. Download manually from: https://www.kaggle.com/datasets/adityajn105/flickr8k (you need to sign up a kaggle account)
2. Place `flickr8k.zip` in the project root directory
3. Rerun the script

The dataset is approximately 1 GB. After extraction and splitting:
- Database: ~7,991 images (for building FAISS index)
- Queries: 100 images (completely separate, for testing)

## Prerequisites

- Ubuntu 20.04 or 22.04 EC2 instance
- Minimum 30 GB storage (50 GB for GPU instances)
- Git and Python 3.8+

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Peter1426/CS4296_2026_Project.git
cd CS4296_2026_Project

# Run the benchmark (downloads dataset, installs dependencies, runs tests)
chmod +x scripts/run_benchmark.sh

# Run CPU benchmark
./scripts/run_benchmark.sh false

# Run GPU benchmark (on GPU instances)
./scripts/run_benchmark.sh true
