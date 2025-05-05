"""
LRU Cache implementation for prediction results
"""
from collections import OrderedDict
import numpy as np
from PIL import Image
import io

class PredictionCache:
    """LRU Cache for storing prediction results"""
    
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.cache = OrderedDict()
        
    def _generate_key(self, image):
        """Generate a cache key from image data"""
        # Convert image to bytes for hashing
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return hash(img_byte_arr.getvalue())
        
    def get(self, image):
        """Get prediction result from cache"""
        key = self._generate_key(image)
        if key in self.cache:
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None
        
    def put(self, image, prediction, confidence, probabilities):
        """Add prediction result to cache"""
        key = self._generate_key(image)
        
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
            
        self.cache[key] = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities
        }
        
    def clear(self):
        """Clear the cache"""
        self.cache.clear()