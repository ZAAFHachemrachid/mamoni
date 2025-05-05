import pytest
from PIL import Image
import numpy as np
from src.utils.prediction_cache import PredictionCache

@pytest.fixture
def cache():
    return PredictionCache(capacity=2)

@pytest.fixture
def sample_image():
    # Create a simple 10x10 test image
    return Image.fromarray(np.zeros((10, 10), dtype=np.uint8))

@pytest.fixture
def sample_prediction():
    return {
        'prediction': 5,
        'confidence': 0.95,
        'probabilities': np.array([0.01] * 9 + [0.95])
    }

class TestPredictionCache:
    def test_cache_initialization(self, cache):
        """Test cache is properly initialized with given capacity"""
        assert cache.capacity == 2
        assert len(cache.cache) == 0

    def test_cache_put_and_get(self, cache, sample_image, sample_prediction):
        """Test basic cache put and get operations"""
        cache.put(sample_image, **sample_prediction)
        result = cache.get(sample_image)
        
        assert result is not None
        assert result['prediction'] == sample_prediction['prediction']
        assert result['confidence'] == sample_prediction['confidence']
        np.testing.assert_array_equal(result['probabilities'], sample_prediction['probabilities'])

    def test_cache_miss(self, cache, sample_image):
        """Test cache miss returns None"""
        assert cache.get(sample_image) is None

    def test_cache_lru_eviction(self, cache, sample_prediction):
        """Test least recently used item is evicted when cache is full"""
        image1 = Image.fromarray(np.ones((10, 10), dtype=np.uint8))
        image2 = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
        image3 = Image.fromarray(np.full((10, 10), 2, dtype=np.uint8))

        # Add 3 items to cache with capacity 2
        cache.put(image1, **sample_prediction)
        cache.put(image2, **sample_prediction)
        cache.put(image3, **sample_prediction)

        # First item should be evicted
        assert cache.get(image1) is None
        assert cache.get(image2) is not None
        assert cache.get(image3) is not None

    def test_cache_update_access_order(self, cache, sample_prediction):
        """Test accessing an item moves it to the end of LRU order"""
        image1 = Image.fromarray(np.ones((10, 10), dtype=np.uint8))
        image2 = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
        
        cache.put(image1, **sample_prediction)
        cache.put(image2, **sample_prediction)
        
        # Access image1 to move it to most recently used
        cache.get(image1)
        
        # Add new image - should evict image2 instead of image1
        image3 = Image.fromarray(np.full((10, 10), 2, dtype=np.uint8))
        cache.put(image3, **sample_prediction)
        
        assert cache.get(image1) is not None
        assert cache.get(image2) is None
        assert cache.get(image3) is not None

    def test_cache_clear(self, cache, sample_image, sample_prediction):
        """Test cache clear operation"""
        cache.put(sample_image, **sample_prediction)
        assert len(cache.cache) == 1
        
        cache.clear()
        assert len(cache.cache) == 0
        assert cache.get(sample_image) is None

    def test_cache_key_generation(self, cache):
        """Test different images generate different cache keys"""
        image1 = Image.fromarray(np.ones((10, 10), dtype=np.uint8))
        image2 = Image.fromarray(np.zeros((10, 10), dtype=np.uint8))
        
        key1 = cache._generate_key(image1)
        key2 = cache._generate_key(image2)
        
        assert key1 != key2