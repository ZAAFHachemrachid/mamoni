import pytest
import asyncio
import os
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from src.utils.model_manager import ModelManager, ModelLoadError
from src.utils.progress import ProgressManager

@pytest.fixture
def model_manager():
    progress_manager = Mock(spec=ProgressManager)
    return ModelManager(progress_manager=progress_manager)

@pytest.fixture
def sample_model_data():
    return {
        'weights': [np.random.rand(10, 5).tolist(), np.random.rand(5, 2).tolist()],
        'biases': [np.random.rand(5).tolist(), np.random.rand(2).tolist()],
        'metadata': {
            'layer_sizes': [10, 5, 2],
            'epochs': 10,
            'learning_rate': 0.01,
            'batch_size': 32
        }
    }

@pytest.fixture
def temp_model_file(tmp_path, sample_model_data):
    model_dir = tmp_path / "configs" / "model_configs"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "test_model.json"
    with open(model_path, 'w') as f:
        json.dump(sample_model_data, f)
    return model_path

@pytest.fixture
def temp_npy_file(tmp_path):
    """Create a temporary .npy model file for testing"""
    model_dir = tmp_path / "configs" / "model_configs"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "test_model.npy"
    
    weights = [np.random.rand(10, 5), np.random.rand(5, 2)]
    biases = [np.random.rand(5), np.random.rand(2)]
    np.save(model_path, [weights, biases])
    return model_path

@pytest.mark.asyncio
async def test_json_model_progress_sequence(model_manager, temp_model_file):
    """Test that JSON model loading yields correct progress sequence"""
    progress_values = []
    final_yield = None
    
    async for item in model_manager.load_model_async(temp_model_file.name):
        if isinstance(item, float):
            progress_values.append(item)
        else:
            final_yield = item
    
    # Verify progress sequence
    assert progress_values[0] == pytest.approx(0.1)  # Initial progress
    assert progress_values[1] == pytest.approx(0.3)  # JSON loaded
    assert progress_values[2] == pytest.approx(0.5)  # Network initialized
    assert progress_values[-2] == pytest.approx(1.0)  # Final progress
    
    # Verify final yield format
    assert isinstance(final_yield, tuple)
    model, metadata = final_yield
    assert len(metadata['layer_sizes']) == 3
    assert all(isinstance(w, np.ndarray) for w in model.weights)

@pytest.mark.asyncio
async def test_npy_model_progress_sequence(model_manager, temp_npy_file):
    """Test that NPY model loading yields correct progress sequence"""
    progress_values = []
    final_yield = None
    
    async for item in model_manager.load_model_async(temp_npy_file.name):
        if isinstance(item, float):
            progress_values.append(item)
        else:
            final_yield = item
    
    # Verify progress sequence for .npy loading
    assert len(progress_values) == 2  # Should yield 0.9 and then final tuple
    assert progress_values[0] == pytest.approx(0.9)
    
    # Verify final yield format
    assert isinstance(final_yield, tuple)
    model, metadata = final_yield
    assert isinstance(metadata, dict)  # Empty dict for legacy format
    assert len(model.weights) == 2

@pytest.mark.asyncio
async def test_error_yield_sequence(model_manager, tmp_path):
    """Test yield behavior during error conditions"""
    invalid_path = tmp_path / "configs" / "model_configs" / "invalid.json"
    invalid_path.parent.mkdir(parents=True)
    
    # Create invalid JSON file
    with open(invalid_path, 'w') as f:
        f.write("invalid json content")
    
    progress_values = []
    with pytest.raises(ModelLoadError) as exc_info:
        async for item in model_manager.load_model_async(invalid_path.name):
            if isinstance(item, float):
                progress_values.append(item)
    
    # Verify we get initial progress before error
    assert len(progress_values) == 1
    assert progress_values[0] == pytest.approx(0.1)
    assert "Failed to load model" in str(exc_info.value)

@pytest.mark.asyncio
async def test_corrupt_npy_yield_sequence(model_manager, temp_npy_file):
    """Test yield sequence with corrupted .npy file"""
    # Corrupt the .npy file
    with open(temp_npy_file, 'wb') as f:
        f.write(b"corrupted data")
    
    progress_values = []
    with pytest.raises(ModelLoadError) as exc_info:
        async for item in model_manager.load_model_async(temp_npy_file.name):
            if isinstance(item, float):
                progress_values.append(item)
    
    assert len(progress_values) == 0  # Should fail before any progress yields
    assert "Failed to load .npy model" in str(exc_info.value)

@pytest.mark.asyncio
async def test_load_model_async_progress(model_manager, temp_model_file):
    """Test async model loading with progress tracking"""
    progress_values = []
    async for progress in model_manager.load_model_async(temp_model_file.name):
        progress_values.append(progress)
        
    assert len(progress_values) >= 4  # Should have multiple progress updates
    assert progress_values[0] == pytest.approx(0.1)  # Initial progress
    assert progress_values[-1] > 0.9  # Final progress should be near 1
    
    # Verify model loaded successfully
    model, metadata = await model_manager.load_model_async(temp_model_file.name).__anext__()
    assert model is not None
    assert metadata['layer_sizes'] == [10, 5, 2]

@pytest.mark.asyncio
async def test_npy_model_loading_progress(model_manager, temp_npy_file):
    """Test progress tracking during .npy model loading"""
    progress_values = []
    
    # Track progress during .npy loading
    async for progress in model_manager.load_model_async(temp_npy_file.name):
        progress_values.append(progress)
        await asyncio.sleep(0)  # Allow other tasks to run
    
    # Verify progress points
    assert len(progress_values) > 0
    assert progress_values[-1] == pytest.approx(0.9)  # NPY loading yields 0.9 before completion
    
    # Verify final model
    model, metadata = await model_manager.load_model_async(temp_npy_file.name).__anext__()
    assert model is not None
    assert len(model.weights) == 2
    assert isinstance(metadata, dict)

@pytest.mark.asyncio
async def test_progress_value_ranges(model_manager, temp_model_file):
    """Test that progress values stay within valid range"""
    async for progress in model_manager.load_model_async(temp_model_file.name):
        assert 0 <= progress <= 1.0, f"Progress value {progress} outside valid range"
        assert isinstance(progress, float), "Progress must be float"

@pytest.mark.asyncio
async def test_npy_loading_error_handling(model_manager, tmp_path):
    """Test error handling during .npy model loading"""
    # Create invalid .npy file
    invalid_path = tmp_path / "configs" / "model_configs" / "invalid.npy"
    invalid_path.parent.mkdir(parents=True)
    invalid_path.write_bytes(b"invalid data")
    
    with pytest.raises(ModelLoadError, match="Failed to load .npy model"):
        async for _ in model_manager.load_model_async(invalid_path.name):
            pass

@pytest.mark.asyncio
async def test_npy_model_cleanup(model_manager, temp_npy_file):
    """Test cleanup after loading .npy model"""
    # Mock np.load to simulate error
    with patch('numpy.load', side_effect=Exception("Simulated error")):
        with pytest.raises(ModelLoadError):
            async for _ in model_manager.load_model_async(temp_npy_file.name):
                pass
    
    # Verify status is updated with error
    assert "Error:" in model_manager.get_load_status()
    
    # Verify can still load models after error
    model_manager.current_load_status = None
    async for progress in model_manager.load_model_async(temp_npy_file.name):
        assert progress >= 0
    
    model, _ = await model_manager.load_model_async(temp_npy_file.name).__anext__()
    assert model is not None

@pytest.mark.asyncio
async def test_error_handling(model_manager):
    """Test error handling during model loading"""
    with pytest.raises(ModelLoadError, match="No model files found"):
        async for _ in model_manager.load_model_async("nonexistent.json"):
            pass

@pytest.mark.asyncio
async def test_event_loop_cleanup(model_manager, temp_model_file):
    """Test proper event loop cleanup"""
    old_loop = asyncio.get_event_loop()
    
    # Load model in new loop
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    
    try:
        model, metadata = await model_manager.load_model_async(temp_model_file.name).__anext__()
        assert model is not None
    finally:
        new_loop.close()
        asyncio.set_event_loop(old_loop)

@pytest.mark.asyncio
async def test_load_legacy_npy_model(model_manager, tmp_path):
    """Test loading legacy .npy model format"""
    # Create test .npy file
    model_dir = tmp_path / "configs" / "model_configs"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "legacy_model.npy"
    
    weights = [np.random.rand(10, 5), np.random.rand(5, 2)]
    biases = [np.random.rand(5), np.random.rand(2)]
    np.save(model_path, [weights, biases])
    
    # Test loading
    model, metadata = await model_manager._load_npy_model_async(model_path)
    assert model is not None
    assert len(model.weights) == 2
    assert all(isinstance(w, np.ndarray) for w in model.weights)

@pytest.mark.asyncio
async def test_concurrent_loading(model_manager, temp_model_file):
    """Test concurrent model loading attempts"""
    async def load_model():
        model, metadata = await model_manager.load_model_async(temp_model_file.name).__anext__()
        return model is not None

    # Try loading model concurrently
    results = await asyncio.gather(
        load_model(),
        load_model(),
        load_model()
    )
    
    assert all(results)  # All loads should succeed

def test_sync_load_fallback(model_manager, temp_model_file):
    """Test synchronous loading fallback"""
    model, metadata = model_manager.load_model(temp_model_file.name)
    assert model is not None
    assert metadata['layer_sizes'] == [10, 5, 2]

@pytest.mark.asyncio
async def test_load_status_tracking(model_manager, temp_model_file):
    """Test load status tracking"""
    assert model_manager.get_load_status() is None
    
    async for _ in model_manager.load_model_async(temp_model_file.name):
        status = model_manager.get_load_status()
        assert status == "Loading model..."
    
    model, _ = await model_manager.load_model_async(temp_model_file.name).__anext__()
    assert model is not None