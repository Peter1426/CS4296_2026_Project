import time
import numpy as np
import psutil
import os
import sys
import csv
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # Add project root directory to Python's import path

from src.feature_extractor import MobileNetFeatureExtractor
from src.index_builder import FAISSIndexBuilder

class BenchmarkRunner:
    # Initialize Benchmark Runner
    def __init__(self):
        self.extractor = MobileNetFeatureExtractor()
        self.results = []
    
    # Run benchmark on current instance
    def run_benchmark(self, dataset_dir, query_images, name="test", k=10):
        print(f"\n{'='*50}")
        print(f"Running benchmark: {name}")
        print(f"{'='*50}")
        
        instance_info = self._get_instance_info()
        
        print("\n[Step 1] Extracting features from dataset...")
        features, image_paths = self.extractor.extract_from_directory(
            dataset_dir, max_images=5000
        )
        
        print("\n[Step 2] Building FAISS index...")
        index_builder = FAISSIndexBuilder()
        build_time = index_builder.build_flat_index(features, image_paths)
        
        print("\n[Step 3] Measuring memory usage...")
        memory_mb = self._get_memory_mb()
        
        print(f"\n[Step 4] Running {len(query_images)} queries...")
        query_times = []
        query_success = 0
        
        # Loop through each query image
        for i, query_path in enumerate(query_images):
            if not os.path.exists(query_path):
                print(f"  Skipping {query_path} (not found)")
                continue
            
            try:
                query_features = self.extractor.extract_from_path(query_path)
                results, search_time = index_builder.search(query_features, k)
                query_times.append(search_time)
                query_success += 1
                
                # Print progress evey 50 images
                if (i + 1) % 50 == 0:
                    print(f"  Completed {i + 1}/{len(query_images)} queries")
                    
            except Exception as e:
                print(f"  Error on {query_path}: {e}")
        
        # Calculate statistics
        avg_query_time = np.mean(query_times) if query_times else 0
        qps = 1.0 / avg_query_time if avg_query_time > 0 else 0
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_name': name,
            'instance_type': instance_info['instance_type'],
            'availability_zone': instance_info['availability_zone'],
            'dataset_size': len(image_paths),
            'num_queries': query_success,
            'build_time_sec': build_time,
            'memory_mb': memory_mb,
            'avg_query_time_ms': avg_query_time * 1000,
            'queries_per_second': qps,
            'k': k
        }
        
        self.results.append(results)
        
        print(f"\n{'='*50}")
        print("Benchmark Complete!")
        print(f"  Dataset size: {len(image_paths)} images")
        print(f"  Build time: {build_time:.2f} sec")
        print(f"  Memory: {memory_mb:.1f} MB")
        print(f"  Avg query time: {avg_query_time*1000:.2f} ms")
        print(f"  QPS: {qps:.2f}")
        print(f"{'='*50}\n")
        
        return results
    
    # Get AWS EC2 instance metadata (using the IMDSv2 method)
    def _get_instance_info(self):
        info = {
            'instance_type': 'unknown',
            'availability_zone': 'unknown'
        }
        
        try:
            import urllib.request
            
            # Get a token (PUT request)
            token_req = urllib.request.Request(
                'http://169.254.169.254/latest/api/token',
                method='PUT',
                headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'}
            )
            with urllib.request.urlopen(token_req, timeout=1) as response:
                token = response.read().decode('utf-8')
            
            # Use token to get instance type
            type_req = urllib.request.Request(
                'http://169.254.169.254/latest/meta-data/instance-type',
                headers={'X-aws-ec2-metadata-token': token}
            )
            with urllib.request.urlopen(type_req, timeout=1) as response:
                info['instance_type'] = response.read().decode('utf-8')
            
            # Use token to get availability zone
            ava_zone_req = urllib.request.Request(
                'http://169.254.169.254/latest/meta-data/placement/availability-zone',
                headers={'X-aws-ec2-metadata-token': token}
            )
            with urllib.request.urlopen(ava_zone_req, timeout=1) as response:
                info['availability_zone'] = response.read().decode('utf-8')
                
        except Exception:
            pass
        
        return info
    
    # Get current memory usage in MB
    def _get_memory_mb(self):
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    # Save results in CSV and JSON format
    def save_results(self, output_file):
        # Check results validity
        if not self.results:
            print("No results to save")
            return
        
        # Write CSV file
        csv_file = output_file.replace('.json', '.csv') if output_file.endswith('.json') else output_file + '.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        
        # Write JSON file
        json_file = output_file if output_file.endswith('.json') else output_file + '.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {csv_file} and {json_file}")

# CLI for benchmark
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark image retrieval on EC2')            # Create ArgumentParser Object
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')           # add a mandatory argument 
    parser.add_argument('--queries', required=True, help='Path to queries directory or file')   # add a mandatory argument
    parser.add_argument('--output', default='benchmark_results.json', help='Output file')       # add an optional argument
    parser.add_argument('--name', default=None, help='Benchmark name')                          # add an optional argument
    parser.add_argument('--k', type=int, default=10, help='Number of results per query')        # add an optional argument
    
    args = parser.parse_args()
    
    # Collect query images
    query_images = []
    if os.path.isdir(args.queries):
        for f in os.listdir(args.queries):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                query_images.append(os.path.join(args.queries, f))
    else:
        query_images.append(args.queries)
    
    # Set benchmark name
    benchmark_name = args.name
    if not benchmark_name:
        try:
            import urllib.request
            # Try to get instance type from AWS metadata
            req = urllib.request.Request(
                'http://169.254.169.254/latest/meta-data/instance-type',
                headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'}
            )
            with urllib.request.urlopen(req, timeout=1) as response:
                benchmark_name = response.read().decode('utf-8')
        except:
            benchmark_name = "unknown_instance"
    
    runner = BenchmarkRunner()
    runner.run_benchmark(args.dataset, query_images, name=benchmark_name, k=args.k)
    runner.save_results(args.output)


if __name__ == "__main__":
    main()