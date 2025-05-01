"""
Prediction Management Module

This module provides utilities for managing predictions using the neural network model.
"""
import numpy as np
from PIL import Image

class PredictionManager:
    """Manages predictions for drawn digits using the neural network model."""

    def __init__(self, parent, prediction_label, controller, dataset):
        """
        Initialize the PredictionManager.

        Args:
            parent: Parent widget
            prediction_label: Label widget to display prediction
            controller: NeuralNetController instance
            dataset: Dataset instance containing feature preparation methods
        """
        self.parent = parent
        self.prediction_label = prediction_label
        self.controller = controller
        self.dataset = dataset

    def predict(self, image):
        """
        Make a prediction on the given image.

        Args:
            image: PIL Image object to predict

        Returns:
            int: Predicted digit, or None if prediction fails
        """
        try:
            if self.controller is None or self.controller.model is None:
                print("No model loaded for prediction")
                return None

            # Convert image to feature format
            features = self._prepare_image_features(image)
            if features is None:
                return None

            # Make prediction
            activations = self.controller.model.forward(features)
            prediction = np.argmax(activations[-1])
            confidence = float(activations[-1][0][prediction])

            # Update prediction label if available
            if self.prediction_label:
                self.prediction_label.config(
                    text=f"Predicted: {prediction} (Confidence: {confidence:.2f})"
                )

            return prediction

        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None

    def _prepare_image_features(self, image):
        """
        Prepare image features for prediction.

        Args:
            image: PIL Image object to process

        Returns:
            numpy.ndarray: Prepared features array, or None if preparation fails
        """
        try:
            # Convert to grayscale numpy array
            img_array = np.array(image.convert('L'))

            # Apply same feature preparation as training data
            if hasattr(self.dataset, 'current_feature_size'):
                feature_size = self.dataset.current_feature_size
                img_array = self._resize_and_prepare(img_array, feature_size)

            # Reshape for model input
            features = img_array.reshape(1, -1)
            return features

        except Exception as e:
            print(f"Error preparing image features: {str(e)}")
            return None

    def _resize_and_prepare(self, img_array, feature_size):
        """
        Resize and prepare image array to match training data format.

        Args:
            img_array: numpy.ndarray of image
            feature_size: tuple of (height, width) for resizing

        Returns:
            numpy.ndarray: Prepared array matching training data format
        """
        try:
            # Convert to PIL Image for resizing
            img = Image.fromarray(img_array)
            img = img.resize(feature_size, Image.Resampling.LANCZOS)
            
            # Convert back to numpy array
            img_array = np.array(img)

            # Apply same preprocessing as training data
            if hasattr(self.dataset, 'prepare_features'):
                img_array = self.dataset.prepare_features(
                    np.expand_dims(img_array, axis=0)
                )[0]

            return img_array

        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            return None