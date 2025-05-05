"""
Prediction Tab Module - Implements the dedicated prediction interface
"""
import customtkinter as ctk
from PIL import Image
import numpy as np
import asyncio
from functools import partial
import time
import threading
import tkinter as tk
from ..preview import ImagePreviewFrame
from utils.prediction import PredictionManager
from utils.prediction_cache import PredictionCache
from ..canvas import CanvasManager

class PredictionTab(ctk.CTkFrame):
    """Prediction interface as a Tab"""
    
    def __init__(self, parent, nn_controller, nn_dataset):
        super().__init__(parent)
        self.parent = parent
        self.controller = nn_controller
        self.dataset = nn_dataset
        
        # State flags
        self.is_model_loaded = False
        self.is_predicting = False
        self.retry_count = 0
        self.max_retries = 3
        
        # Prediction cache and history
        self.prediction_cache = PredictionCache(capacity=100)
        self.prediction_history = []
        self.max_history = 5
        
        # Confidence thresholds
        self.confidence_threshold_low = 0.4
        self.confidence_threshold_high = 0.8
        
        # Initialize canvas
        self.canvas_manager = CanvasManager(self)
        self.canvas_manager.stroke_callback = self.on_drawing_update
        
        self._create_layout()
        
    def _create_layout(self):
        """Create the main layout"""
        # Left side - Controls and Probability Bars
        left_frame = ctk.CTkFrame(self)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Model loading status
        self.loading_label = ctk.CTkLabel(left_frame, text="Loading model...", text_color="gray")
        self.loading_label.pack(padx=5, pady=5)
        
        self.loading_progress = ctk.CTkProgressBar(left_frame)
        self.loading_progress.pack(padx=5, pady=(0,10), fill="x")
        self.loading_progress.set(0)
        
        # Model control buttons
        buttons_frame = ctk.CTkFrame(left_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)
        
        self.load_model_btn = ctk.CTkButton(buttons_frame, text="Load Model", command=self._load_model, state="normal")
        self.load_model_btn.pack(side="left", padx=2)
        
        clear_btn = ctk.CTkButton(buttons_frame, text="Clear", command=self._clear_canvas)
        clear_btn.pack(side="right", padx=2)

        # Title for probability section
        self.prob_label = ctk.CTkLabel(left_frame, text="Prediction Probabilities")
        self.prob_label.pack(padx=5, pady=(15,5))
        self.prob_label.pack_forget()  # Hide until model is loaded
        
        # Create probability bars with improved styling
        prob_container = ctk.CTkFrame(left_frame)
        prob_container.pack(fill="x", padx=10, pady=5)
        
        self.prob_bars = []
        self.prob_labels = []
        
        for i in range(10):
            # Container for each probability row
            row_frame = ctk.CTkFrame(prob_container)
            row_frame.pack(fill="x", padx=5, pady=2)
            
            # Label showing digit and probability
            label = ctk.CTkLabel(row_frame, text=f"Digit {i}: 0%", width=100)
            label.pack(side="left", padx=5)
            self.prob_labels.append(label)
            label.pack_forget()
            
            # Progress bar showing probability
            bar = ctk.CTkProgressBar(row_frame, height=20)
            bar.pack(side="right", fill="x", expand=True, padx=5)
            bar.set(0)
            self.prob_bars.append(bar)
            bar.pack_forget()

        # Preview section
        preview_label = ctk.CTkLabel(left_frame, text="Processed Image")
        preview_label.pack(padx=5, pady=(15,5))
        
        self.preview = ImagePreviewFrame(left_frame)
        self.preview.pack(padx=5, pady=5)
        
        # History section
        history_label = ctk.CTkLabel(left_frame, text="Recent Predictions")
        history_label.pack(padx=5, pady=(15,5))
        
        self.history_frame = ctk.CTkFrame(left_frame)
        self.history_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Right side - Canvas
        right_frame = ctk.CTkFrame(self)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        canvas_label = ctk.CTkLabel(right_frame, text="Draw a digit")
        canvas_label.pack(padx=5, pady=5)
        
        self.canvas_manager.canvas.pack(expand=True, padx=5, pady=5)
        
        
        # Initialize prediction manager
        self.prediction_manager = PredictionManager(self, None, self.controller, self.dataset)
        self._check_model_loaded()
        
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
            
    def _check_model_loaded(self):
        """Check if the model is loaded and update UI accordingly"""
        is_loaded = self.controller and self.controller.is_model_loaded()
        
        if is_loaded != self.is_model_loaded:
            self.is_model_loaded = is_loaded
            if is_loaded:
                self.loading_label.configure(text="Model ready", text_color="green")
                self.loading_progress.set(1)
                self.prob_label.pack()
                for label, bar in zip(self.prob_labels, self.prob_bars):
                    label.pack()
                    bar.pack()
            else:
                self.loading_label.configure(text="No model loaded", text_color="gray")
                self.loading_progress.set(0)
                self.prob_label.pack_forget()
                for label, bar in zip(self.prob_labels, self.prob_bars):
                    label.pack_forget()
                    bar.pack_forget()
        
        self.after(1000, self._check_model_loaded)
            
    def on_drawing_update(self, image):
        """Callback for canvas drawing updates"""
        if not self.is_predicting and self.is_model_loaded:
            # Run prediction asynchronously
            asyncio.run(self.predict(image))
            
    async def predict(self, image):
        """Make a prediction on the given image with caching and retries"""
        if image is None or self.is_predicting:
            return None, None
            
        self.is_predicting = True
        self.preview.display_image(image)
        self._update_prediction_status("Processing...", "gray")
        
        try:
            # Check cache first
            cache_result = self.prediction_cache.get(image)
            if cache_result:
                self._update_prediction_status("Retrieved from cache", "blue")
                self.update_probabilities(cache_result['probabilities'])
                self.add_to_history(image, cache_result['prediction'], cache_result['confidence'])
                return cache_result['prediction'], cache_result['confidence']
            
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    if self.controller and self.controller.model:
                        # Show prediction progress
                        self._update_prediction_status("Running neural network...", "orange")
                        
                        # Process image and get prediction
                        prediction = self.controller.process_canvas_input(image)
                        
                        # Get processed features using same preprocessing as prediction
                        feature_method = getattr(self.controller.dataset, 'feature_method', 'average')
                        feature_size = getattr(self.controller.dataset, 'current_feature_size', (5, 5))
                        
                        # Convert and normalize image
                        img_array = np.array(image) / 255.0
                        
                        # Extract features using dataset's method
                        features = self.controller.dataset.image_to_features(
                            img_array,
                            method=feature_method,
                            feature_size=feature_size
                        )
                        features = features.reshape(1, -1)
                        
                        # Get probabilities using processed features
                        activations = self.controller.model.forward(features)
                        probabilities = activations[-1]
                        confidence = float(probabilities[0][prediction])
                        
                        # Cache the result
                        self.prediction_cache.put(image, prediction, confidence, probabilities)
                        
                        # Update display with confidence indicators
                        self._update_prediction_status(
                            self._get_confidence_message(confidence),
                            self._get_confidence_color(confidence)
                        )
                        self.update_probabilities(probabilities)
                        self.add_to_history(image, prediction, confidence)
                        
                        return prediction, confidence
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count < self.max_retries:
                        self._update_prediction_status(f"Retry {retry_count}/{self.max_retries}...", "orange")
                        await asyncio.sleep(0.5)  # Wait before retry
                    else:
                        self._update_prediction_status(f"Error: {str(e)}", "red")
                        raise
            
        except Exception as e:
            self._update_prediction_status(f"Prediction failed: {str(e)}", "red")
            return None, None
            
        finally:
            self.is_predicting = False
            
        return None, None
        
    def _update_prediction_status(self, message, color):
        """Update the prediction status display"""
        self.loading_label.configure(text=message, text_color=color)
        
    def _get_confidence_message(self, confidence):
        """Get appropriate message based on confidence level"""
        if confidence >= self.confidence_threshold_high:
            return f"High confidence: {confidence:.2%}"
        elif confidence >= self.confidence_threshold_low:
            return f"Medium confidence: {confidence:.2%}"
        else:
            return f"Low confidence: {confidence:.2%}"
            
    def _get_confidence_color(self, confidence):
        """Get appropriate color based on confidence level"""
        if confidence >= self.confidence_threshold_high:
            return "green"
        elif confidence >= self.confidence_threshold_low:
            return "orange"
        else:
            return "red"
        
    def _clear_canvas(self):
        """Clear the canvas and reset prediction displays"""
        self.canvas_manager.reset()
        self.preview.clear()
        
        # Reset probability displays
        for i in range(10):
            self.prob_bars[i].set(0)
            self.prob_labels[i].configure(text=f"Digit {i}: 0%")
            
        # Clear history and reset status
        self.prediction_history.clear()
        self.update_history_display()
        self._update_prediction_status("Ready", "green" if self.is_model_loaded else "gray")

    async def _load_model_async(self):
        """Load model asynchronously"""
        try:
            self.load_model_btn.configure(state="disabled")
            self.loading_label.configure(text="Loading model...", text_color="gray")
            self.loading_progress.set(0)
            
            # Hide probability displays
            self.prob_label.pack_forget()
            for label, bar in zip(self.prob_labels, self.prob_bars):
                label.pack_forget()
                bar.pack_forget()
                
            # Create model loading generator
            model_loader = self.controller.model_manager.load_model_async()
            
            # Track progress updates
            model = None
            metadata = {}
            async for progress in model_loader:
                self.loading_progress.set(progress)
                # Use after() to update UI in main thread
                self.after(0, lambda p=progress: self.loading_progress.set(p))
                await asyncio.sleep(0)  # Allow UI updates
                
            # Get final result
            model, metadata = await model_loader.__anext__()
            self.controller.model = model
            
            self.is_model_loaded = True
            self.loading_label.configure(text="Model ready", text_color="green")
            self.loading_progress.set(1)
            
            # Show probability displays in main thread
            self.after(0, lambda: [
                self.prob_label.pack(),
                *[label.pack() for label in self.prob_labels],
                *[bar.pack() for bar in self.prob_bars]
            ])
        except Exception as e:
            self.loading_label.configure(text=f"Error: {str(e)}", text_color="red")
            self.loading_progress.set(0)
        finally:
            self.load_model_btn.configure(state="normal")
            
    def _load_model(self):
        """Trigger asynchronous model loading"""
        if self.controller:
            # Reset UI state
            self.is_model_loaded = False
            
            async def load_with_proactor():
                # Use IocpProactor event loop for Windows
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    await self._load_model_async()
                finally:
                    loop.close()
            
            # Run in thread to avoid blocking UI
            thread = threading.Thread(
                target=lambda: asyncio.run(load_with_proactor())
            )
            thread.daemon = True
            thread.start()