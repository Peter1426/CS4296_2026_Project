import faiss
import numpy as np
import os
import time
import pickle

class FAISSIndexBuilder:
    # Initialize FAISS index builder
    def __init__(self, dimension=1280, use_gpu=False):
        self.dimension = dimension
        self.index = None
        self.image_paths = []
        self.use_gpu = use_gpu
        self.gpu_available = False
        self.num_gpus = 0

        # Chekc gpu validity
        if use_gpu:
            print("Checking GPU availability...")
            try:
                self.num_gpus = faiss.get_num_gpus()
                self.gpu_available = self.num_gpus > 0
                if self.gpu_available:
                    print(f"GPU mode enabled. Found {self.num_gpus} GPU(s) available")
                else:
                    print("Warning: No GPU detected, falling back to CPU mode")
                    self.use_gpu = False

            except Exception as e:
                print(f"Warning: GPU initialization failed: {e}")
                print("Falling back to CPU mode")
                self.use_gpu = False
    
    # Build the FlatL2 index
    def build_flat_index(self, features, image_paths):
        print(f"Building FlatL2 index with {len(features)} vectors...")
        start_time = time.time()
        
        self.index = faiss.IndexFlatL2(self.dimension)  # Create FlatL2 index
        self.index.add(features.astype('float32'))      # Add feature vectors to index in float32 data type
        self.image_paths = image_paths                  # Store list of image paths
        
        # Check if using gpu mode
        if self.use_gpu and self.gpu_available:
            print(f"Converting index to multi-GPU mode using {self.num_gpus} GPU(s)...")
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        
        # Show details of index
        elapsed = time.time() - start_time
        print(f"Index built in {elapsed:.2f} seconds")
        print(f"Index size: {self.index.ntotal} vectors")
        print(f"Memory usage: {self._estimate_memory_mb():.1f} MB")
        
        return elapsed
    
    # Search for k most similar images
    def search(self, query_vector, k=10):
        # Check index validity
        if self.index is None:
            raise ValueError("Index not built. Call build_flat_index() first.")
        
        query = query_vector.reshape(1, -1).astype('float32')   # Convert vector to a 2D array with a single row
        
        start_time = time.time()
        distances, indices = self.index.search(query, k)    # Use FAISS's built-in search function 
        search_time = time.time() - start_time
        
        # Save the result of the FAISS search
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.image_paths) and idx >= 0:
                results.append({
                    'image_path': self.image_paths[idx],
                    'distance': float(distances[0][i]),
                    'rank': i + 1
                })
        
        return results, search_time
    
    # Save index and image paths to disk
    def save(self, save_dir):
        # Check index validity
        if self.index is None:
            raise ValueError("No index to save")
        
        # Create directiry if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert to cpu before saving
        if self.use_gpu and self.gpu_available:
            print("Converting GPU index to CPU for saving...")
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            index_path = os.path.join(save_dir, 'faiss_index.bin')      # Create the full path for the FAISS index file
            faiss.write_index(cpu_index, index_path)                   # Use FAISS's built-in function to save index to disk
        else:
            index_path = os.path.join(save_dir, 'faiss_index.bin')      
            faiss.write_index(self.index, index_path)

        # Save the image paths file in pickle (byte stream)
        paths_path = os.path.join(save_dir, 'image_paths.pkl')
        with open(paths_path, 'wb') as f:
            pickle.dump(self.image_paths, f)
        
        print(f"Index saved to {save_dir}")
    
    # Load index and image paths from disk
    def load(self, save_dir):
        index_path = os.path.join(save_dir, 'faiss_index.bin')
        paths_path = os.path.join(save_dir, 'image_paths.pkl')
        
        self.index = faiss.read_index(index_path)   # Use FAISS's built-in function to read index from disk

        # Load the image paths from pickle
        with open(paths_path, 'rb') as f:
            self.image_paths = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors from {save_dir}")
    
    # Estimate memory usage of index
    def _estimate_memory_mb(self):
        # Check index validity
        if self.index is None:
            return 0
        
        # Calculate approximate memory usage (Multiply by 4 since each float32 is 4 bytes)
        bytes_used = self.index.ntotal * self.dimension * 4
        return bytes_used / (1024 * 1024)