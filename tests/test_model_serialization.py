import sys
import os
import numpy as np
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_layer import NeuralNetwork, PrefillSettings, PrefillMethod
from src.data.data_layer import ImageDataset

def test_model_serialization():
    """Test saving and loading model in both JSON and pickle formats."""
    
    # Create a simple model
    print("\nCreating test model...")
    layer_sizes = [2, 3, 1]  # Simple network: 2 inputs, 3 hidden, 1 output
    original_model = NeuralNetwork(layer_sizes)
    
    # Save in both formats
    print("\nSaving model in both formats...")
    json_path = "test_model.json"
    pkl_path = "test_model.pkl"
    
    original_model.save_model(json_path)
    original_model.save_model(pkl_path)
    
    # Load both formats
    print("\nLoading models from both formats...")
    json_model = NeuralNetwork(layer_sizes)
    pkl_model = NeuralNetwork(layer_sizes)
    
    json_model.load_model(json_path)
    pkl_model.load_model(pkl_path)
    
    # Verify parameters match
    print("\nVerifying model parameters...")
    
    def verify_parameters(model1, model2, format_name):
        weights_match = all(
            np.allclose(w1, w2) 
            for w1, w2 in zip(model1.weights, model2.weights)
        )
        biases_match = all(
            np.allclose(b1, b2) 
            for b1, b2 in zip(model1.biases, model2.biases)
        )
        print(f"{format_name} format verification:")
        print(f"- Weights match: {weights_match}")
        print(f"- Biases match: {biases_match}")
        return weights_match and biases_match
    
    json_success = verify_parameters(original_model, json_model, "JSON")
    pkl_success = verify_parameters(original_model, pkl_model, "Pickle")
    
    # Clean up test files
    print("\nCleaning up test files...")
    try:
        os.remove(json_path)
        os.remove(pkl_path)
        print("Test files removed successfully")
    except Exception as e:
        print(f"Error cleaning up files: {e}")
    
    # Final result
    if json_success and pkl_success:
        print("\nTest PASSED: Both formats successfully saved and loaded!")
    else:
        print("\nTest FAILED: Inconsistencies found in model parameters")

def test_prefill_settings_serialization():
    """Test PrefillSettings serialization and validation."""
    print("\n=== Testing PrefillSettings Serialization ===")
    
    # Test creation with valid values
    settings = PrefillSettings(enabled=True, method=PrefillMethod.AVERAGE, size=5)
    assert settings.enabled == True, "Enabled flag not set correctly"
    assert settings.method == PrefillMethod.AVERAGE, "Method not set correctly"
    assert settings.size == 5, "Size not set correctly"
    print("Basic initialization: PASSED")

    # Test serialization to dict
    settings_dict = settings.to_dict()
    assert settings_dict["enabled"] == True
    assert settings_dict["method"] == "average"
    assert settings_dict["size"] == 5
    print("Dictionary serialization: PASSED")

    # Test deserialization from dict
    new_settings = PrefillSettings.from_dict(settings_dict)
    assert new_settings.enabled == settings.enabled
    assert new_settings.method == settings.method
    assert new_settings.size == settings.size
    print("Dictionary deserialization: PASSED")

    # Test validation
    try:
        invalid_settings = PrefillSettings(enabled=True, method="invalid", size=5)
        assert False, "Should have raised ValueError for invalid method"
    except ValueError:
        print("Invalid method validation: PASSED")

    try:
        invalid_settings = PrefillSettings(enabled=True, method=PrefillMethod.AVERAGE, size=11)
        assert False, "Should have raised ValueError for invalid size"
    except ValueError:
        print("Invalid size validation: PASSED")

def test_pooling_methods():
    """Test different pooling methods in the ImageDataset class."""
    print("\n=== Testing Pooling Methods ===")
    
    # Create test image (8x8 checkerboard pattern)
    test_image = np.zeros((8, 8))
    test_image[::2, ::2] = 255  # Set alternate pixels to create pattern
    test_image[1::2, 1::2] = 255
    
    dataset = ImageDataset()
    
    # Test average pooling
    avg_features = dataset._average_pooling(test_image, (4, 4))
    assert avg_features.shape == (16,), "Average pooling output shape incorrect"
    assert np.all(avg_features == 127.5), "Average pooling values incorrect"
    print("Average pooling test: PASSED")
    
    # Test max pooling
    max_features = dataset._max_pooling(test_image, (4, 4))
    assert max_features.shape == (16,), "Max pooling output shape incorrect"
    assert np.all(max_features == 255), "Max pooling values incorrect"
    print("Max pooling test: PASSED")
    
    # Test sum pooling
    sum_features = dataset._sum_pooling(test_image, (4, 4))
    assert sum_features.shape == (16,), "Sum pooling output shape incorrect"
    assert np.all(sum_features == 510), "Sum pooling values incorrect"
    print("Sum pooling test: PASSED")
    
def test_prefill_validation():
    """Test validation of prefill settings."""
    print("\n=== Testing Prefill Validation ===")
    
    # Test method validation
    settings = PrefillSettings()
    
    try:
        settings.method = "invalid"
        assert False, "Should have raised ValueError for invalid method string"
    except ValueError as e:
        print("Invalid method string validation: PASSED")
        
    try:
        settings.method = 123
        assert False, "Should have raised ValueError for invalid method type"
    except ValueError as e:
        print("Invalid method type validation: PASSED")
        
    # Test size validation
    try:
        settings.size = 2
        assert False, "Should have raised ValueError for size < 3"
    except ValueError as e:
        print("Size lower bound validation: PASSED")
        
    try:
        settings.size = 11
        assert False, "Should have raised ValueError for size > 10"
    except ValueError as e:
        print("Size upper bound validation: PASSED")
        
    try:
        settings.size = "5"
        assert False, "Should have raised ValueError for non-integer size"
    except ValueError as e:
        print("Size type validation: PASSED")
    
    # Test enabled validation
    try:
        settings.enabled = "true"
        assert False, "Should have raised ValueError for non-boolean enabled"
    except ValueError as e:
        print("Enabled type validation: PASSED")

if __name__ == "__main__":
    test_model_serialization()
    test_prefill_settings_serialization()
    test_pooling_methods()
    test_prefill_validation()