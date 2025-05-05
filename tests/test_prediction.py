import pytest
import asyncio
import threading
import tkinter as tk
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from PIL import Image
import customtkinter as ctk
from src.ui.tabs.prediction import PredictionTab
from src.utils.model_manager import ModelManager
from src.core.neural_network import NeuralNetwork

class AsyncMockController:
    def __init__(self):
        self.model = None
        self.model_manager = Mock(spec=ModelManager)
        self.model_manager.load_model_async = AsyncMock()

@pytest.fixture
def prediction_tab(qtbot):
    root = ctk.CTk()
    controller = AsyncMockController()
    dataset = Mock()
    tab = PredictionTab(root, controller, dataset)
    yield tab
    root.destroy()

@pytest.fixture
def mock_model():
    model = Mock(spec=NeuralNetwork)
    model.forward.return_value = [np.array([[0.1] * 10])]  # Mock probabilities
    return model

@pytest.mark.asyncio
async def test_ui_updates_during_loading(prediction_tab, mock_model):
    """Test UI updates during model loading"""
    # Mock the model loading process
    progress_updates = [0.1, 0.3, 0.5, 0.8, 1.0]
    prediction_tab.controller.model_manager.load_model_async.return_value = (
        AsyncMockGenerator(progress_updates, (mock_model, {}))
    )

    # Start loading
    load_thread = threading.Thread(
        target=lambda: asyncio.run(prediction_tab._load_model_async())
    )
    load_thread.start()
    load_thread.join()

    # Verify UI updates occurred
    assert prediction_tab.loading_progress.get() == 1.0
    assert prediction_tab.is_model_loaded
    assert prediction_tab.loading_label.cget("text") == "Model ready"
    assert prediction_tab.loading_label.cget("text_color") == "green"

@pytest.mark.asyncio
async def test_thread_safe_progress_updates(prediction_tab):
    """Test thread-safe progress bar updates"""
    updates_received = []
    
    def progress_callback(value):
        updates_received.append(value)
    
    # Monitor progress updates
    prediction_tab.loading_progress.set = progress_callback
    
    async def simulate_progress():
        for progress in [0.2, 0.4, 0.6, 0.8, 1.0]:
            prediction_tab.after(0, lambda p=progress: 
                prediction_tab.loading_progress.set(p))
            await asyncio.sleep(0.1)
    
    # Run progress updates
    await simulate_progress()
    
    assert len(updates_received) == 5
    assert updates_received[-1] == 1.0

@pytest.mark.asyncio
@patch('asyncio.get_event_loop')
async def test_windows_proactor_integration(mock_get_loop, prediction_tab, mock_model):
    """Test Windows IocpProactor event loop integration"""
    mock_loop = AsyncMock()
    mock_get_loop.side_effect = RuntimeError  # Simulate no existing loop
    
    # Start model loading
    prediction_tab._load_model()
    
    # Wait for thread to complete
    await asyncio.sleep(0.5)
    
    # Verify new loop was created
    assert prediction_tab.is_model_loaded is False  # Should be updated when load completes
    mock_get_loop.assert_called()

@pytest.mark.asyncio
async def test_prediction_error_handling(prediction_tab, mock_model):
    """Test error handling during prediction"""
    prediction_tab.controller.model = mock_model
    prediction_tab.is_model_loaded = True
    
    # Simulate error during prediction
    mock_model.forward.side_effect = Exception("Network error")
    
    # Create test image
    test_image = Image.new('L', (28, 28), color='white')
    
    # Attempt prediction
    prediction, confidence = await prediction_tab.predict(test_image)
    
    # Verify error handling
    assert prediction is None
    assert confidence is None
    assert "Error" in prediction_tab.loading_label.cget("text")
    assert prediction_tab.loading_label.cget("text_color") == "red"

@pytest.mark.asyncio
async def test_prediction_retry_mechanism(prediction_tab, mock_model):
    """Test prediction retry mechanism"""
    prediction_tab.controller.model = mock_model
    prediction_tab.is_model_loaded = True
    
    # Fail twice then succeed
    fail_count = 0
    def mock_forward(x):
        nonlocal fail_count
        if fail_count < 2:
            fail_count += 1
            raise Exception("Temporary error")
        return [np.array([[0.1] * 10])]
    
    mock_model.forward.side_effect = mock_forward
    
    # Create test image
    test_image = Image.new('L', (28, 28), color='white')
    
    # Attempt prediction
    prediction, confidence = await prediction_tab.predict(test_image)
    
    # Verify retries worked
    assert prediction is not None
    assert fail_count == 2
    assert prediction_tab.loading_label.cget("text_color") != "red"

@pytest.mark.asyncio
async def test_ui_cleanup_after_prediction(prediction_tab, mock_model):
    """Test UI cleanup after prediction completes"""
    prediction_tab.controller.model = mock_model
    prediction_tab.is_model_loaded = True
    
    # Create test image
    test_image = Image.new('L', (28, 28), color='white')
    
    # Track cleanup
    initial_state = prediction_tab.is_predicting
    
    # Run prediction
    await prediction_tab.predict(test_image)
    
    # Verify cleanup
    assert not prediction_tab.is_predicting
    assert initial_state != prediction_tab.is_predicting
    
    # Verify probability bars were updated
    for bar in prediction_tab.prob_bars:
        assert bar.get() is not None

class AsyncMockGenerator:
    """Helper class to simulate async generator for model loading"""
    def __init__(self, progress_values, final_result):
        self.progress_values = progress_values
        self.final_result = final_result
        self.current = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.current < len(self.progress_values):
            value = self.progress_values[self.current]
            self.current += 1
            return value
        else:
            return self.final_result

if __name__ == '__main__':
    pytest.main([__file__])