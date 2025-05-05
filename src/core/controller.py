import numpy as np
from .neural_network import NeuralNetwork
from src.utils.model_manager import ModelManager, ModelLoadError
from src.utils.progress import ProgressManager, ProgressState
from PIL import Image
import asyncio

class NeuralNetController:
    """Handles training and evaluation of neural network models."""
    
    def __init__(self, dataset=None, progress_manager=None):
        self.dataset = dataset
        self.model = None
        self.progress_manager = progress_manager
        self.model_manager = ModelManager(progress_manager)
        self.current_state = ProgressState.IDLE
        self.prediction_callback = None
        self._last_prediction = None
        self._last_confidence = None
        # Training parameters
        self.initial_learning_rate = 0.01
        self.current_learning_rate = self.initial_learning_rate
        self.min_learning_rate = 1e-6
        self.lr_decay_rate = 0.95
        self.lr_decay_steps = 1000
        self.batch_size = 256
        self.steps = 0
        
    def set_dataset(self, dataset):
        """Set the dataset to use for training/testing."""
        self.dataset = dataset

    def set_prediction_callback(self, callback):
        """Set callback for when predictions are made."""
        self.prediction_callback = callback
        
    def create_model(self, layer_sizes=None, input_size=None):
        """Create a new neural network model with improved architecture."""
        try:
            if layer_sizes is None:
                if input_size is None:
                    if self.dataset and self.dataset.features is not None:
                        input_size = self.dataset.features.shape[1]
                    else:
                        raise ValueError("Cannot create model: no dataset loaded or layer_sizes specified")
                output_size = self.dataset.num_classes
                # Updated architecture [2500->128->64->output]
                layer_sizes = [input_size] + [128, 64] + [output_size]
            
            if self.progress_manager:
                self.progress_manager.start(text="Creating new model...")
                
            self.model = NeuralNetwork(layer_sizes)
            
            if self.progress_manager:
                self.progress_manager.stop(text="Model created successfully")
            
            return self.model
            
        except Exception as e:
            if self.progress_manager:
                self.progress_manager.error(str(e))
            raise

    async def load_model_async(self, filepath=None):
        """Load model asynchronously with progress tracking."""
        try:
            if self.progress_manager:
                self.progress_manager.start(text="Loading model...")
            
            self.model, metadata = await self.model_manager.load_model_async(filepath)
            
            if self.progress_manager:
                self.progress_manager.stop(text="Model loaded successfully")
                
            return metadata
            
        except ModelLoadError as e:
            if self.progress_manager:
                self.progress_manager.error(str(e))
            raise
            
    def process_canvas_input(self, image: Image.Image):
        """Process canvas input and make prediction."""
        try:
            if self.model is None:
                raise ValueError("No model loaded")
            
            if self.dataset is None:
                raise ValueError("No dataset loaded to get preprocessing parameters")
                
            # Convert image to numpy array and normalize
            img_array = np.array(image) / 255.0
            
            # Apply same feature extraction as training data
            feature_method = getattr(self.dataset, 'feature_method', 'average')
            feature_size = getattr(self.dataset, 'current_feature_size', (5, 5))
            
            # Use dataset's image_to_features method for consistency
            features = self.dataset.image_to_features(img_array, method=feature_method, feature_size=feature_size)
            features = features.reshape(1, -1)  # Add batch dimension
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            self._last_prediction = prediction
            
            # Notify callback if set
            if self.prediction_callback:
                self.prediction_callback(prediction)
                
            return prediction
            
        except Exception as e:
            if self.progress_manager:
                self.progress_manager.error(str(e))
            raise

    def train_epoch(self, learning_rate=None, batch_size=None):
        """Train the model for one epoch with learning rate decay and improved metrics."""
        try:
            if self.model is None:
                raise ValueError("No model initialized. Call create_model first.")

            if self.dataset is None or self.dataset.train_features is None:
                raise ValueError("No dataset loaded or features not prepared and split.")

            if self.progress_manager:
                self.progress_manager.start(mode='determinate', text="Training epoch...")

            # Use instance parameters if not provided
            if learning_rate is None:
                learning_rate = self.current_learning_rate
            if batch_size is None:
                batch_size = self.batch_size
                
            total_batches = len(self.dataset.train_features) // batch_size
            
            # Apply learning rate decay
            self.steps += total_batches
            self.current_learning_rate = max(
                self.initial_learning_rate * (self.lr_decay_rate ** (self.steps / self.lr_decay_steps)),
                self.min_learning_rate
            )
            
            loss = self.model.train_epoch(
                self.dataset.train_features,
                self.dataset.train_labels,
                lr=self.current_learning_rate,
                batch_size=batch_size,
                progress_callback=lambda batch:
                    self.progress_manager.update(batch, total_batches,
                    f"Training batch {batch}/{total_batches} (lr={self.current_learning_rate:.6f})") if self.progress_manager else None
            )

            # Calculate training metrics
            predictions = self.model.predict(self.dataset.train_features)
            true_labels = np.argmax(self.dataset.train_labels, axis=1)
            accuracy = np.mean(predictions == true_labels)

            if self.progress_manager:
                self.progress_manager.stop(text=f"Epoch complete - Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")

            return loss, accuracy

        except Exception as e:
            if self.progress_manager:
                self.progress_manager.error(str(e))
            raise

    def validate(self):
        """Validate current model on validation dataset."""
        try:
            if self.model is None:
                raise ValueError("No model initialized. Call create_model first.")

            if self.dataset is None or self.dataset.val_features is None:
                raise ValueError("No dataset loaded or features not prepared and split.")

            if self.progress_manager:
                self.progress_manager.start(text="Validating model...")

            # Get predictions and confidence scores
            activations = self.model.forward(self.dataset.val_features)[-1]
            predictions = np.argmax(activations, axis=1)
            true_labels = np.argmax(self.dataset.val_labels, axis=1)
            
            # Calculate accuracy and confidence metrics
            accuracy = np.mean(predictions == true_labels)
            confidence_scores = np.max(activations, axis=1)
            mean_confidence = np.mean(confidence_scores)
            
            # Calculate validation loss with improved numerical stability
            log_probs = np.log(activations + 1e-8)
            val_loss = -np.sum(self.dataset.val_labels * log_probs) / len(predictions)
            
            # Store validation metrics
            self._last_confidence = mean_confidence

            if self.progress_manager:
                self.progress_manager.stop(text=f"Validation complete - Loss: {val_loss:.4f}, Accuracy: {accuracy:.2%}")

            return val_loss, accuracy

        except Exception as e:
            if self.progress_manager:
                self.progress_manager.error(str(e))
            raise

    def evaluate(self):
        """Evaluate current model on test dataset."""
        try:
            if self.model is None:
                raise ValueError("No model initialized. Call create_model first.")

            if self.dataset is None or self.dataset.test_features is None:
                raise ValueError("No dataset loaded or features not prepared and split.")

            if self.progress_manager:
                self.progress_manager.start(text="Evaluating model...")

            predictions = self.model.predict(self.dataset.test_features)
            accuracy = np.mean(predictions == np.argmax(self.dataset.test_labels, axis=1))

            if self.progress_manager:
                self.progress_manager.stop(text=f"Evaluation complete - Accuracy: {accuracy:.2%}")

            return accuracy

        except Exception as e:
            if self.progress_manager:
                self.progress_manager.error(str(e))
            raise

    def save_model(self, filepath):
        """Save model to file with progress tracking."""
        try:
            if self.model is None:
                raise ValueError("No model to save. Train or load a model first.")
                
            if self.progress_manager:
                self.progress_manager.start(text="Saving model...")
                
            self.model_manager.save_model(self.model)
            
            if self.progress_manager:
                self.progress_manager.stop(text="Model saved successfully")
                
        except Exception as e:
            if self.progress_manager:
                self.progress_manager.error(str(e))
            raise

    def get_last_prediction(self):
        """Get the most recent prediction result with confidence."""
        return self._last_prediction, self._last_confidence

    def is_model_loaded(self):
        """Check if a model is currently loaded."""
        return self.model is not None

    def get_current_state(self):
        """Get the current state of the controller."""
        if self.progress_manager:
            return self.progress_manager.get_state()
        return self.current_state