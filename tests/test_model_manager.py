import pytest
import os
import json
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime
from src.utils.model_manager import ModelManager
from src.core.neural_network import NeuralNetwork

@pytest.fixture(autouse=True)
def clean_models_dir():
    """Automatically clean up models directory before and after each test."""
    models_dir = ModelManager.MODELS_DIR
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir()
    yield
    if models_dir.exists():
        shutil.rmtree(models_dir)

@pytest.fixture
def model_manager():
    """Create a ModelManager instance."""
    return ModelManager()

@pytest.fixture
def sample_network():
    """Create a sample neural network for testing."""
    model = NeuralNetwork(layer_sizes=[784, 128, 10])
    # Initialize with some random weights for testing
    for w, b in zip(model.weights, model.biases):
        w[:] = np.random.randn(*w.shape)
        b[:] = np.random.randn(*b.shape)
    return model

class TestModelNaming:
    def test_name_generation_components(self, model_manager):
        """Test if generated name includes all required components."""
        layer_sizes = [784, 128, 10]
        dataset_name = "mnist"
        
        name = model_manager.generate_model_name(layer_sizes, dataset_name)
        
        # Check name components
        assert dataset_name in name
        assert "784x128x10" in name
        assert datetime.now().strftime("%Y%m%d") in name

    def test_name_uniqueness(self, model_manager):
        """Test if different architectures generate unique names."""
        architectures = [
            ([784, 128, 10], "mnist"),
            ([784, 256, 10], "mnist"),
            ([784, 128, 10], "cifar")
        ]
        
        names = [model_manager.generate_model_name(*arch) for arch in architectures]
        assert len(set(names)) == len(names), "Generated names should be unique"

    def test_name_consistency(self, model_manager):
        """Test if same parameters generate consistent name structure."""
        name1 = model_manager.generate_model_name([784, 128, 10], "mnist")
        name2 = model_manager.generate_model_name([784, 128, 10], "mnist")
        
        # Names should be different (due to timestamp) but follow same pattern
        assert name1 != name2
        assert name1.split("-")[:-1] == name2.split("-")[:-1]

class TestModelSavingLoading:
    def test_save_model_structure(self, model_manager, sample_network):
        """Test if model saving creates correct directory structure."""
        save_path = model_manager.save_model(sample_network, "test_dataset")
        save_dir = Path(save_path)
        
        assert save_dir.exists()
        assert (save_dir / "parameters.npy").exists()
        assert (save_dir / "metadata.json").exists()

    def test_save_load_parameters(self, model_manager, sample_network):
        """Test if model parameters are preserved after save and load."""
        # Save model
        save_path = model_manager.save_model(sample_network, "test_dataset")
        model_name = Path(save_path).name

        # Load model
        loaded_model, _ = model_manager.load_model(model_name)

        # Compare parameters
        for w1, w2 in zip(sample_network.weights, loaded_model.weights):
            assert np.allclose(w1, w2)
        for b1, b2 in zip(sample_network.biases, loaded_model.biases):
            assert np.allclose(b1, b2)

    def test_metadata_preservation(self, model_manager, sample_network):
        """Test if metadata is correctly saved and loaded."""
        metadata = {
            "test_accuracy": 0.95,
            "training_epochs": 10
        }
        
        # Save with metadata
        save_path = model_manager.save_model(sample_network, "test_dataset", metadata)
        model_name = Path(save_path).name
        
        # Load and verify metadata
        _, loaded_metadata = model_manager.load_model(model_name)
        assert loaded_metadata["test_accuracy"] == metadata["test_accuracy"]
        assert loaded_metadata["training_epochs"] == metadata["training_epochs"]
        assert "architecture" in loaded_metadata
        assert "layer_sizes" in loaded_metadata["architecture"]

    def test_invalid_model_error(self, model_manager):
        """Test error handling for invalid model types."""
        with pytest.raises(TypeError):
            model_manager.save_model({"invalid": "model"}, "test_dataset")

    def test_missing_model_error(self, model_manager):
        """Test error handling for loading non-existent model."""
        with pytest.raises(FileNotFoundError):
            model_manager.load_model("non_existent_model")

class TestModelListing:
    def test_list_models_structure(self, model_manager, sample_network):
        """Test if list_models returns correct structure."""
        # Save multiple models
        model_manager.save_model(sample_network, "dataset1")
        model_manager.save_model(sample_network, "dataset2")
        
        models = model_manager.list_models()
        
        assert len(models) == 2
        for model in models:
            assert "name" in model
            assert "metadata" in model
            assert "architecture" in model["metadata"]
            assert "dataset" in model["metadata"]

    def test_metadata_retrieval_accuracy(self, model_manager, sample_network):
        """Test accuracy of metadata in listed models."""
        metadata = {"test_key": "test_value"}
        save_path = model_manager.save_model(sample_network, "test_dataset", metadata)
        model_name = Path(save_path).name
        
        models = model_manager.list_models()
        found_model = next(m for m in models if m["name"] == model_name)
        
        assert found_model["metadata"]["test_key"] == "test_value"
        assert found_model["metadata"]["dataset"] == "test_dataset"