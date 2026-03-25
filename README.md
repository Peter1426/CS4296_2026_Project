# CS4296_2026_Project
This is a course project from CS4296 at year 2026

The project aims to measure and compare the performance of an image retrieval system over different Amazon EC2 instances.

The image retrieval system consists of two components:
- **Feature extraction**: MobileNetV2 converts images to 1280-dimensional feature vectors
- **Similarity search**: FAISS (Facebook AI Similarity Search) performs efficient vector search

The system is deployed and benchmarked on various EC2 instance types (t3.medium, c5.large, r5.large) to evaluate:
- Query latency and throughput (QPS)
- Index construction time
- Memory usage
- Cost-performance trade-offs

The project uses the **Flickr8k** dataset (8,092 images). The dataset is automatically downloaded during setup and split into:
- **Database**: ~7,992 images (for building the search index)
- **Queries**: 100 images (completely separate, used for testing)

## Running the Benchmark

```bash
# Clone the repository
git clone https://github.com/Peter1426/CS4296_2026_Project.git
cd CS4296_2026_Project

# Run the benchmark (downloads dataset, installs dependencies, runs tests)
chmod +x scripts/run_benchmark.sh
./scripts/run_benchmark.sh