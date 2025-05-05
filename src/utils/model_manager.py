import os
import json
import numpy as np
from pathlib import Path
import asyncio
from ..core.neural_network import NeuralNetwork
from .progress import ProgressManager, ProgressState

class ModelLoadError(Exception):
    """Exception raised for model loading errors."""
    pass

class ModelManager:
    """Handles automated model saving and loading."""
    
    MODEL_DIR = Path("configs/model_configs")
    
    def __init__(self, progress_manager=None):
        """Initialize ModelManager and ensure save directory exists."""
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.progress_manager = progress_manager
        self.current_load_status = None
        
    def generate_model_name(self, layer_sizes, epochs=None, learning_rate=None, batch_size=None):
        """Generate descriptive model filename."""
        layers_str = '_'.join(map(str, layer_sizes))
        
        components = [f"best_{layers_str}"]
        if epochs is not None:
            components.append(f"{epochs}e")
        if learning_rate is not None:
            lr_str = f"{learning_rate:g}".replace(".", "")
            components.append(f"{lr_str}lr")
        if batch_size is not None:
            components.append(f"{batch_size}bs")
            
        return '_'.join(components) + ".json"

    def save_model(self, model, epochs=None, learning_rate=None, batch_size=None):
        """Save model with descriptive filename."""
        if not isinstance(model, NeuralNetwork):
            raise TypeError("Model must be an instance of NeuralNetwork")

        # Get layer sizes from model
        layer_sizes = [model.weights[0].shape[1]] + [w.shape[0] for w in model.weights]
        
        # Generate descriptive filename
        filename = self.generate_model_name(layer_sizes, epochs, learning_rate, batch_size)
        save_path = self.MODEL_DIR / filename
        
        # Save model parameters as json
        model_data = {
            'weights': [w.tolist() for w in model.weights],
            'biases': [b.tolist() for b in model.biases],
            'metadata': {
                'layer_sizes': layer_sizes,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size
            }
        }
        
        try:
            with open(save_path, 'w') as f:
                json.dump(model_data, f)
        except Exception as e:
            raise ModelLoadError(f"Failed to save model: {str(e)}")

    async def load_model_async(self, filename=None):
        """Load model asynchronously with progress tracking.
        
        Args:
            filename (str, optional): Specific model file to load. If None, loads latest.
            
        Yields:
            float: Progress value between 0 and 1
            
        Returns:
            tuple: (NeuralNetwork, dict) - Loaded model and metadata
        """
        try:
            if filename is None:
                # Get most recent .json file
                json_files = list(self.MODEL_DIR.glob("*.json"))
                if not json_files:
                    # Try .npy files for backwards compatibility
                    npy_files = list(self.MODEL_DIR.glob("*.npy"))
                    if not npy_files:
                        raise ModelLoadError("No model files found")
                    filepath = sorted(npy_files, key=lambda x: x.stat().st_mtime)[-1]
                    self.current_load_status = "Loading legacy model format..."
                    if self.progress_manager:
                        await self.progress_manager.track_async_task(
                            self._load_npy_model_async(filepath),
                            description="Loading legacy model..."
                        )
                    model = await self._load_npy_model_async(filepath)
                    yield 0.9  # Almost done
                    yield (model, {})  # Final yield
                filepath = sorted(json_files, key=lambda x: x.stat().st_mtime)[-1]
            else:
                filepath = self.MODEL_DIR / filename
                if not filepath.exists():
                    raise ModelLoadError(f"Model file not found: {filepath}")
            
            # Load based on file extension
            if filepath.suffix == '.npy':
                self.current_load_status = "Loading legacy model format..."
                model = await self._load_npy_model_async(filepath)
                yield 0.9  # Almost done
                yield (model, {})  # Final yield for legacy format
            else:
                self.current_load_status = "Loading model..."
                yield 0.1  # Initial progress
                
                # Load and parse JSON
                with open(filepath) as f:
                    data = json.load(f)
                yield 0.3  # JSON loaded
                
                layer_sizes = data['metadata']['layer_sizes']
                model = NeuralNetwork(layer_sizes)
                yield 0.5  # Network initialized
                
                # Convert weights/biases to numpy arrays with progress updates
                total_arrays = len(data['weights']) + len(data['biases'])
                for i, w in enumerate(data['weights']):
                    model.weights[i] = np.array(w)
                    yield 0.5 + ((i + 1) / total_arrays) * 0.25
                    
                for i, b in enumerate(data['biases']):
                    model.biases[i] = np.array(b)
                    yield 0.75 + ((i + 1) / total_arrays) * 0.25
                
                yield 1.0  # Complete
                yield (model, data['metadata'])  # Final yield with metadata
                
        except Exception as e:
            self.current_load_status = f"Error: {str(e)}"
            raise ModelLoadError(f"Failed to load model: {str(e)}")

    async def _load_npy_model_async(self, filepath):
        """Load model from legacy .npy format asynchronously."""
        try:
            # Load model data
            await asyncio.sleep(0)  # Allow other tasks to run
            model_data = np.load(filepath, allow_pickle=True)
            
            # Extract weights and compute layer sizes
            weights = model_data[0]
            layer_sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]
            
            # Initialize and load model
            model = NeuralNetwork(layer_sizes)
            model.load_model(filepath)
            
            await asyncio.sleep(0)  # Another yield point
            return model
        except Exception as e:
            raise ModelLoadError(f"Failed to load .npy model: {str(e)}")

    async def _load_json_model_async(self, filepath):
        """Load model from .json format asynchronously."""
        try:
            # Simulate async load for large files
            await asyncio.sleep(0)  # Allow other tasks to run
            
            with open(filepath) as f:
                data = json.load(f)
                
            layer_sizes = data['metadata']['layer_sizes']
            model = NeuralNetwork(layer_sizes)
            
            # Convert lists back to numpy arrays
            model.weights = [np.array(w) for w in data['weights']]
            model.biases = [np.array(b) for b in data['biases']]
            
            return model, data['metadata']
        except Exception as e:
            raise ModelLoadError(f"Failed to load JSON model: {str(e)}")

    def get_load_status(self):
        """Get current model loading status."""
        return self.current_load_status

    # Synchronous fallback methods for compatibility
    def load_model(self, filename=None):
        """Synchronous version of load_model for backwards compatibility."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.load_model_async(filename))
        finally:
            loop.close()