import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Disable TensorFlow oneDNN custom operations warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # Hide CUDA devices

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import time

class MobileNetFeatureExtractor:
    # Load pre-trained MobileNetV2 model
    def __init__(self):
        print("Loading MobileNetV2 model...")
        self.model = MobileNetV2(
            weights='imagenet',         # Load pre-trained weights on ImageNet dataset
            include_top=False,          # Remove the classification layer at top
            pooling='avg',               # Applies global average pooling to output 1280-dimensional vector
            input_shape=(224, 224, 3)   # Explicitly define the input shape
        )
        self.input_size = (224, 224)    # Resize image to 224*224 pixels
        print("Model loaded successfully")
    
    # Extract feature from a single image
    def extract_from_path(self, img_path):
        img = image.load_img(img_path, target_size=self.input_size)
        
        x = image.img_to_array(img)                     # Convert image to array
        x = np.expand_dims(x, axis=0)                   # Add a batch dimension to the array
        x = preprocess_input(x)                         # Scales pixel value to [-1, 1] for MobileNetV2 to process
        features = self.model.predict(x, verbose=0)     # Get the features of the preprocess image from MobileNetV2

        return features.flatten()                       # Flatten the feature to a 1D array
    
    # Extract features from multiple images
    def extract_batch(self, image_paths, verbose=True):
        features_list = []
        valid_paths = []
        
        # Loop through each image path
        for i, img_path in enumerate(image_paths):
            try:
                features = self.extract_from_path(img_path)
                features_list.append(features)
                valid_paths.append(img_path)
                
                # Print progress evey 100 images
                if verbose and (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(image_paths)} images")

            # Error handling
            except Exception as e:
                print(f"  Warning: Failed to process {img_path}: {e}")
        
        return np.array(features_list).astype('float32'), valid_paths   # Conver list to numpy array for FAISS later
    
    # Extract features from all images inside a directory
    def extract_from_directory(self, image_dir, max_images=None, extensions=('.jpg', '.jpeg', '.png')):
        image_paths = []
        for filename in os.listdir(image_dir):

            # Check if file extension is in the allowed list
            if filename.lower().endswith(extensions):
                image_paths.append(os.path.join(image_dir, filename))

                # Check if we reach the max limit of number of images
                if max_images and len(image_paths) >= max_images:
                    break
        
        print(f"Found {len(image_paths)} images in {image_dir}")
        
        # Extract feature and save time elapsed
        print("Extracting features...")
        start_time = time.time()
        features, valid_paths = self.extract_batch(image_paths)
        elapsed = time.time() - start_time
        
        print(f"Extracted {len(valid_paths)} features in {elapsed:.2f} seconds")
        print(f"Average: {elapsed/len(valid_paths):.3f} seconds per image")
        
        return features, valid_paths