# CS4296 - Image Retrieval Benchmark on AWS EC2

## Project Overview

This is an individual project from **CS4296 Cloud Computing** at year 2026. This project benchmarks the performance of an image retrieval system over different Amazon EC2 instances.

The image retrieval system consists of two components:
- **Feature extraction**: *MobileNetV2* converts images to 1280-dimensional feature vectors
- **Similarity search**: *FAISS* (Facebook AI Similarity Search) performs efficient vector search with CPU and GPU support

## Key Feature

- Automatic dataset download
    - The project uses the **Flickr8k** dataset (8,092 images)
- Automatic dataset splitting
    - The **Flickr8k** dataset are split into **Database**: 7,992 images and **Queries**: 100 images
- CPU and GPU benchmark modes
- Multi-GPU support (auto-detects)
- Comprehensive performance metrics:
    - Query latency (ms)
    - Query throughput (QPS)
    - Index build time (sec)
    - Memory usage (MB)
- Result saved in JSON an CSV formats

## Evaluate EC2 instances

The system is deployed and benchmarked on various EC2 instance types (t3.medium, c5.large, r5.large, g4dn.xlarge, g4dn.12xlarge) to evaluate.

| Type | Purpose |
|------|---------|
| t3.medium | CPU baseline (general purpose) |
| c5.large | CPU compute-optimized |
| r5.large | CPU memory-optimized |
| g4dn.xlarge | Single GPU (NVIDIA T4) |
| g4dn.12xlarge | Multi-GPU (4× T4) |

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
