import numpy as np
from .neural_network import NeuralNetwork

class NeuralNetController:
    """Handles training and evaluation of neural network models."""
    
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.model = None

    def set_dataset(self, dataset):
        """Set the dataset to use for training/testing."""
        self.dataset = dataset

    def create_model(self, layer_sizes=None, input_size=None):
        """Create a new neural network model with specified architecture."""
        if layer_sizes is None:
            if input_size is None:
                if self.dataset and self.dataset.features is not None:
                    input_size = self.dataset.features.shape[1]
                else:
                    raise ValueError("Cannot create model: no dataset loaded or layer_sizes specified")
            output_size = self.dataset.num_classes
            layer_sizes = [input_size] + layer_sizes + [output_size]
        self.model = NeuralNetwork(layer_sizes)
        return self.model

    def train_epoch(self, learning_rate, batch_size):
        """Train the model for one epoch and return metrics."""
        if self.model is None:
            raise ValueError("No model initialized. Call create_model first.")

        if self.dataset is None or self.dataset.train_features is None:
            raise ValueError("No dataset loaded or features not prepared and split.")

        loss = self.model.train_epoch(
            self.dataset.train_features,
            self.dataset.train_labels,
            lr=learning_rate,
            batch_size=batch_size
        )

        predictions = self.model.predict(self.dataset.train_features)
        accuracy = np.mean(predictions == np.argmax(self.dataset.train_labels, axis=1))

        return loss, accuracy

    def validate(self):
        """Validate current model on validation dataset."""
        if self.model is None:
            raise ValueError("No model initialized. Call create_model first.")

        if self.dataset is None or self.dataset.val_features is None:
            raise ValueError("No dataset loaded or features not prepared and split.")

        predictions = self.model.predict(self.dataset.val_features)
        accuracy = np.mean(predictions == np.argmax(self.dataset.val_labels, axis=1))

        # Calculate cross-entropy loss on validation set
        activations = self.model.forward(self.dataset.val_features)
        log_probs = np.log(activations[-1] + 1e-8)
        val_loss = -np.sum(self.dataset.val_labels.reshape(self.dataset.val_labels.shape[0], -1) * log_probs) / self.dataset.val_labels.shape[0]

        return val_loss, accuracy

    def evaluate(self):
        """Evaluate current model on test dataset."""
        if self.model is None:
            raise ValueError("No model initialized. Call create_model first.")

        if self.dataset is None or self.dataset.test_features is None:
            raise ValueError("No dataset loaded or features not prepared and split.")

        predictions = self.model.predict(self.dataset.test_features)
        accuracy = np.mean(predictions == np.argmax(self.dataset.test_labels, axis=1))

        return accuracy

    def save_model(self, filepath):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        self.model.save_model(filepath)

    def load_model(self, filepath, input_size=None):
        """Load model from file."""
        # Create a dummy model first if none exists
        if self.model is None:
            default_input_size = 25  # Default if no info available
            if input_size is not None:
                default_input_size = input_size
            elif self.dataset and self.dataset.features is not None:
                default_input_size = self.dataset.features.shape[1]
            self.create_model([default_input_size, 10, 10])  # Will be overwritten by loaded model
        self.model.load_model(filepath)