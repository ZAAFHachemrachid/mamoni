"""
Prediction Tab Module - Implements the dedicated prediction interface
"""
import customtkinter as ctk
from PIL import Image
import numpy as np
from ..preview import ImagePreviewFrame
from utils.prediction import PredictionManager

class PredictionTab(ctk.CTkFrame):
    """Prediction interface as a Tab"""
    
    def __init__(self, parent, nn_controller, nn_dataset):
        super().__init__(parent)
        self.parent = parent
        self.controller = nn_controller
        self.dataset = nn_dataset
        
        # Store recent predictions
        self.prediction_history = []
        self.max_history = 5
        
        self._create_layout()
        
    def _create_layout(self):
        """Create the main layout"""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        
        # Left side - Preview and History
        left_frame = ctk.CTkFrame(self)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Preview section
        preview_label = ctk.CTkLabel(left_frame, text="Current Image")
        preview_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.preview = ImagePreviewFrame(left_frame)
        self.preview.grid(row=1, column=0, padx=5, pady=5)
        
        # History section
        history_label = ctk.CTkLabel(left_frame, text="Recent Predictions")
        history_label.grid(row=2, column=0, padx=5, pady=(15,5))
        
        self.history_frame = ctk.CTkFrame(left_frame)
        self.history_frame.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        
        # Right side - Probability Bars
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Title for probability section
        prob_label = ctk.CTkLabel(right_frame, text="Prediction Probabilities")
        prob_label.grid(row=0, column=0, padx=5, pady=5)
        
        # Create probability bars
        self.prob_bars = []
        self.prob_labels = []
        for i in range(10):
            # Label showing digit and probability
            label = ctk.CTkLabel(right_frame, text=f"Digit {i}: 0%")
            label.grid(row=i+1, column=0, padx=5, pady=(2,0), sticky="w")
            self.prob_labels.append(label)
            
            # Progress bar showing probability
            bar = ctk.CTkProgressBar(right_frame)
            bar.grid(row=i+1, column=1, padx=5, pady=(2,0), sticky="ew")
            bar.set(0)
            self.prob_bars.append(bar)
        
        # Initialize prediction manager
        self.prediction_manager = PredictionManager(self, None, self.controller, self.dataset)
        
    def update_probabilities(self, probabilities):
        """Update probability bars with new predictions"""
        for i, prob in enumerate(probabilities[0]):
            self.prob_bars[i].set(float(prob))
            self.prob_labels[i].configure(text=f"Digit {i}: {prob*100:.1f}%")
            
    def add_to_history(self, image, prediction, confidence):
        """Add a new prediction to the history"""
        # Add new prediction to start of list
        self.prediction_history.insert(0, {
            'image': image,
            'prediction': prediction,
            'confidence': confidence
        })
        
        # Trim history to max length
        if len(self.prediction_history) > self.max_history:
            self.prediction_history = self.prediction_history[:self.max_history]
            
        self.update_history_display()
        
    def update_history_display(self):
        """Update the history display"""
        # Clear existing history display
        for widget in self.history_frame.winfo_children():
            widget.destroy()
            
        # Add each history item
        for i, item in enumerate(self.prediction_history):
            frame = ctk.CTkFrame(self.history_frame)
            frame.grid(row=i, column=0, padx=5, pady=2, sticky="ew")
            
            # Create mini preview of the image
            preview = ImagePreviewFrame(frame)
            preview.grid(row=0, column=0, padx=2, pady=2)
            preview.display_image(item['image'])
            
            # Add prediction info
            info = ctk.CTkLabel(
                frame, 
                text=f"Predicted: {item['prediction']}\nConf: {item['confidence']:.2f}"
            )
            info.grid(row=0, column=1, padx=2, pady=2)
            
    def predict(self, image):
        """Make a prediction on the given image"""
        if image is not None:
            self.preview.display_image(image)
            
            # Get prediction and probabilities
            features = self.prediction_manager._prepare_image_features(image)
            if features is not None and self.controller and self.controller.model:
                activations = self.controller.model.forward(features)
                probabilities = activations[-1]
                prediction = np.argmax(probabilities)
                confidence = float(probabilities[0][prediction])
                
                # Update display
                self.update_probabilities(probabilities)
                self.add_to_history(image, prediction, confidence)
                
                return prediction, confidence
        return None, None