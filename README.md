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
    - To speed up the process, only 5000 images of the 7,991 images **Database** will be used. 
- CPU and GPU benchmark modes
- Multi-GPU support (auto-detects)
- Comprehensive performance metrics:
    - Query latency (ms)
    - Query throughput (QPS)
    - Index build time (sec)
    - Memory usage (MB)
- Result saved in JSON an CSV formats

## Evaluate EC2 instances

The system is deployed and benchmarked on six EC2 instance types (t3.micro, t3.medium, t3.large, c5.large, r5.large, m5.large) to evaluate.

| Type | vCPU | RAM | Purpose |
|------|------|-----|---------|
| t3.micro | 2 | 1 GB | CPU baseline (RAM impact test) |
| t3.medium | 2 | 4 GB | CPU baseline (RAM impact test) |
| t3.large | 2 | 8 GB | CPU baseline (RAM impact test) |
| c5.large | 2 | 4 GB | compute-optimized |
| r5.large | 2 | 16 GB | memory-optimized |
| m5.large | 2 | 8 GB | Balanced alternative |

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
- Git and Python 3.10+

## FOLDER STRUCTURE:
```
Program/
├── scripts/
|   └── run_benchmark.sh    # script for benchmark automatically
├── src/
│   ├── benchmark.py    # benchmark method
│   ├── feature_extaction.py    # MobileNetV2 feature extractor
│   └── index_builder.py    # FAISS index builder
├── data/ (auto-created)
│   ├── images/
│   └── queries/
├── results/ (auto-created)
|   ├── benchmark_results.json
│   └── benchmark_results.csv
├── collected_results/
│   ├── Summary_Result.xlsx    # Summary of six instances results
│   ├── c5_large_benchmark_results.json
│   ├── m5_large_benchmark_results.json
│   ├── r5_large_benchmark_results.json
│   ├── t3_large_benchmark_results.json
│   ├── t3_medium_benchmark_results.json
│   └── t3_micro_benchmark_results.json
└── figures
    ├── Test_Result_AvgQueryTime.png
    ├── Test_Result_BuildTime.png
    ├── Test_Result_Memory.png
    ├── Test_Result_QPS.png
    ├── Test_Result_QPS_vs_LinuxCostPerHour.png
    └── Test_Result_QPS_vs_UbuntuCostPerHour.png
```

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
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
