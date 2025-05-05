import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch
from src.utils.prediction import PredictionManager

class MockDataset:
    def __init__(self, feature_size=(28, 28)):
        self.current_feature_size = feature_size
        
    def prepare_features(self, features):
        return features / 255.0

@pytest.fixture
def mock_controller():
    controller = Mock()
    controller.model = Mock()
    return controller

@pytest.fixture
def mock_label():
    return Mock()

@pytest.fixture
def prediction_manager(mock_controller, mock_label):
    dataset = MockDataset()
    return PredictionManager(Mock(), mock_label, mock_controller, dataset)

@pytest.fixture
def sample_image():
    return Image.fromarray(np.zeros((28, 28), dtype=np.uint8))

class TestPredictionManager:
    def test_prediction_with_no_model(self, prediction_manager):
        """Test prediction handling when no model is loaded"""
        prediction_manager.controller.model = None
        result = prediction_manager.predict(sample_image())
        assert result is None

    def test_successful_prediction(self, prediction_manager, sample_image):
        """Test successful prediction with confidence"""
        # Mock model output
        mock_output = np.zeros((1, 10))
        mock_output[0][5] = 0.95  # High confidence for digit 5
        prediction_manager.controller.model.forward.return_value = [mock_output]
        
        result = prediction_manager.predict(sample_image)
        
        assert result == 5
        prediction_manager.prediction_label.config.assert_called_with(
            text="Predicted: 5 (Confidence: 0.95)"
        )

    def test_low_confidence_prediction(self, prediction_manager, sample_image):
        """Test prediction with low confidence"""
        # Mock model output with low confidence
        mock_output = np.full((1, 10), 0.1)  # Equal probabilities
        prediction_manager.controller.model.forward.return_value = [mock_output]
        
        result = prediction_manager.predict(sample_image)
        
        assert result == 0  # First class with equal probability
        prediction_manager.prediction_label.config.assert_called_with(
            text="Predicted: 0 (Confidence: 0.10)"
        )

    def test_feature_preparation_failure(self, prediction_manager):
        """Test handling of feature preparation failure"""
        # Create invalid image
        invalid_image = Mock()
        invalid_image.convert.side_effect = Exception("Image conversion failed")
        
        result = prediction_manager.predict(invalid_image)
        assert result is None

    def test_model_forward_error(self, prediction_manager, sample_image):
        """Test handling of model forward pass error"""
        prediction_manager.controller.model.forward.side_effect = Exception("Forward pass failed")
        
        result = prediction_manager.predict(sample_image)
        assert result is None

    def test_feature_size_adaptation(self, prediction_manager):
        """Test image resizing to match feature size"""
        # Create larger test image
        large_image = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
        
        # Mock model output
        mock_output = np.zeros((1, 10))
        mock_output[0][3] = 0.8
        prediction_manager.controller.model.forward.return_value = [mock_output]
        
        result = prediction_manager.predict(large_image)
        
        # Verify prediction was made after resizing
        assert result == 3
        # Verify forward pass received correctly sized input
        called_features = prediction_manager.controller.model.forward.call_args[0][0]
        assert called_features.shape == (1, 784)  # 28x28 = 784

    def test_feature_preparation_with_dataset_method(self, prediction_manager, sample_image):
        """Test feature preparation using dataset's prepare_features method"""
        mock_output = np.zeros((1, 10))
        mock_output[0][7] = 0.9
        prediction_manager.controller.model.forward.return_value = [mock_output]
        
        result = prediction_manager.predict(sample_image)
        
        assert result == 7
        # Verify forward pass received normalized features (divided by 255)
        called_features = prediction_manager.controller.model.forward.call_args[0][0]
        assert np.max(called_features) <= 1.0

    @patch('PIL.Image.fromarray')
    def test_resize_error_handling(self, mock_fromarray, prediction_manager, sample_image):
        """Test handling of errors during image resizing"""
        mock_fromarray.side_effect = Exception("Resize operation failed")
        
        result = prediction_manager._resize_and_prepare(
            np.zeros((100, 100)), 
            (28, 28)
        )
        assert result is None

    def test_confidence_calculation(self, prediction_manager, sample_image):
        """Test confidence score calculation from model output"""
        # Create varying confidence scenarios
        test_cases = [
            (np.array([[0.9] + [0.01] * 9]), 0, 0.9),  # High confidence
            (np.array([[0.3] * 10]), 0, 0.3),  # Low confidence
            (np.array([[0.0] * 10]), 0, 0.0),  # Zero confidence
        ]
        
        for output, expected_class, expected_confidence in test_cases:
            prediction_manager.controller.model.forward.return_value = [output]
            result = prediction_manager.predict(sample_image)
            
            assert result == expected_class
            # Verify confidence in label text
            label_text = prediction_manager.prediction_label.config.call_args[1]['text']
            assert f"Confidence: {expected_confidence:.2f}" in label_text