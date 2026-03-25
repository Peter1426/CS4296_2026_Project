import faiss
import numpy as np
import os
import time
import pickle

class FAISSIndexBuilder:
    # Initialize FAISS index builder
    def __init__(self, dimension=1280):
        self.dimension = dimension
        self.index = None
        self.image_paths = []
    
    # Build the FlatL2 index
    def build_flat_index(self, features, image_paths):
        print(f"Building FlatL2 index with {len(features)} vectors...")
        start_time = time.time()
        
        self.index = faiss.IndexFlatL2(self.dimension)  # Create FlatL2 index
        self.index.add(features.astype('float32'))      # Add feature vectors to index in float32 data type
        self.image_paths = image_paths                  # Store list of image paths
        
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
        
        index_path = os.path.join(save_dir, 'faiss_index.bin')      # Create the full path for the FAISS index file
        faiss.write_index(self.index, index_path)                   # Use FAISS's built-in function to save index to disk
        
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