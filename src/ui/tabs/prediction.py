"""
Prediction Tab Module - Implements the dedicated prediction interface
"""
import customtkinter as ctk
from PIL import Image, ImageOps
import numpy as np
import asyncio
from functools import partial
import time
import threading
import tkinter as tk
from tkinter import filedialog
from ..canvas import CanvasManager, DEFAULT_BRUSH_RADIUS
from ..preview import ImagePreviewFrame
from utils.prediction import PredictionManager
from utils.prediction_cache import PredictionCache

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
        
        # Canvas manager will be initialized in _create_layout
        
        self._create_layout()
        
    def _create_layout(self):
        """Create the main layout"""
        # Main container with left-right split
        main_container = ctk.CTkFrame(self)
        main_container.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Left side - Predictions and Controls (consistent across tabs)
        left_frame = ctk.CTkFrame(main_container)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        # Right side - Tab control for Draw/Import
        right_frame = ctk.CTkFrame(main_container)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Initialize tab control on right side
        self.tab_view = ctk.CTkTabview(right_frame)
        self.tab_view.pack(expand=True, fill="both")
        
        # Create tabs
        self.draw_tab = self.tab_view.add("Drawing Input")
        self.import_tab = self.tab_view.add("Image Import")
        
        # Add canvas to draw tab
        # Initialize canvas with brush size first
        self.brush_size = ctk.IntVar(value=DEFAULT_BRUSH_RADIUS)
        self.canvas_manager = CanvasManager(self.draw_tab, brush_radius_var=self.brush_size)
        self.canvas_manager.stroke_callback = self.on_drawing_update
        
        # Drawing tools frame
        tools_frame = ctk.CTkFrame(self.draw_tab)
        tools_frame.pack(fill="x", padx=5, pady=5)
        
        # Brush size control
        brush_label = ctk.CTkLabel(tools_frame, text="Brush Size:")
        brush_label.pack(side="left", padx=5)
        
        brush_slider = ctk.CTkSlider(tools_frame,
            from_=5, to=30,
            variable=self.brush_size,
            width=150)
        brush_slider.pack(side="left", padx=5)
        
        # Tool buttons frame
        tool_buttons = ctk.CTkFrame(tools_frame, fg_color="transparent")
        tool_buttons.pack(side="right", padx=5)
        
        # Clear button
        clear_btn = ctk.CTkButton(tool_buttons, text="Clear",
            command=self._clear_canvas,
            width=70)
        clear_btn.pack(side="right", padx=2)
        
        # Undo button
        undo_btn = ctk.CTkButton(tool_buttons, text="Undo",
            command=self.canvas_manager.undo,
            width=70)
        undo_btn.pack(side="right", padx=2)
        
        # Drawing instructions
        instructions = ctk.CTkLabel(self.draw_tab,
            text="Draw a digit in the canvas below\nRight-click to erase strokes",
            text_color="gray")
        instructions.pack(padx=5, pady=5)
        
        # Pack the canvas last
        self.canvas_manager.canvas.pack(expand=True, padx=5, pady=5)
        # Left side layout - Fixed components
        left_title = ctk.CTkLabel(left_frame, text="Neural Network Predictions", font=("Arial", 14, "bold"))
        left_title.pack(padx=5, pady=(0,10))
        
        # Model controls section
        control_section = ctk.CTkFrame(left_frame)
        control_section.pack(fill="x", padx=5, pady=5)
        
        # Model loading status
        self.loading_label = ctk.CTkLabel(control_section, text="Loading model...", text_color="gray")
        self.loading_label.pack(padx=5, pady=5)
        
        self.loading_progress = ctk.CTkProgressBar(control_section)
        self.loading_progress.pack(padx=5, pady=(0,5), fill="x")
        self.loading_progress.set(0)
        
        # Model control buttons
        buttons_frame = ctk.CTkFrame(control_section)
        buttons_frame.pack(fill="x", padx=5, pady=5)
        
        self.load_model_btn = ctk.CTkButton(buttons_frame, text="Load Model", command=self._load_model, state="normal")
        self.load_model_btn.pack(side="left", padx=2)
        
        clear_btn = ctk.CTkButton(buttons_frame, text="Clear Drawing", command=self._clear_canvas)
        clear_btn.pack(side="right", padx=2)

        # Prediction probabilities section
        self.prob_label = ctk.CTkLabel(left_frame, text="Prediction Probabilities", font=("Arial", 12, "bold"))
        self.prob_label.pack(padx=5, pady=(15,5))
        self.prob_label.pack_forget()  # Hide until model is loaded
        
        # Create probability bars with improved styling
        prob_container = ctk.CTkFrame(left_frame, fg_color="transparent")
        prob_container.pack(fill="x", padx=5, pady=5)
        
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

        # Preview and history in scrollable section
        scroll_container = ctk.CTkScrollableFrame(left_frame)
        scroll_container.pack(fill="both", expand=True, padx=5, pady=(10,5))
        
        # Preview section
        preview_label = ctk.CTkLabel(scroll_container, text="Processed Image", font=("Arial", 12, "bold"))
        preview_label.pack(padx=5, pady=(5,5))
        
        self.preview = ImagePreviewFrame(scroll_container)
        self.preview.pack(padx=5, pady=5)
        
        # History section
        history_label = ctk.CTkLabel(scroll_container, text="Recent Predictions", font=("Arial", 12, "bold"))
        history_label.pack(padx=5, pady=(15,5))
        
        self.history_frame = ctk.CTkFrame(scroll_container)
        self.history_frame.pack(fill="both", expand=True, padx=5, pady=5)

        
        # Import tab content
        import_frame = ctk.CTkFrame(self.import_tab)
        import_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        import_label = ctk.CTkLabel(
            import_frame,
            text="Import Image\nSupported formats: PNG, JPEG",
            font=("Arial", 14, "bold")
        )
        import_label.pack(pady=(20,10))
        
        # Buttons frame
        buttons_frame = ctk.CTkFrame(import_frame, fg_color="transparent")
        buttons_frame.pack(pady=10)
        
        import_image_btn = ctk.CTkButton(buttons_frame,
            text="Choose Image File",
            command=self._import_image)
        import_image_btn.pack(side="left", padx=5)
        
        import_csv_btn = ctk.CTkButton(buttons_frame,
            text="Import CSV",
            command=self._import_mnist_csv)
        import_csv_btn.pack(side="left", padx=5)

        # Status label for import
        self.import_status = ctk.CTkLabel(import_frame,
            text="",
            text_color="gray")
        self.import_status.pack(pady=5)

        # Preview for imported image
        preview_label = ctk.CTkLabel(import_frame, text="Preview")
        preview_label.pack(pady=(20,5))
        
        self.import_preview = ImagePreviewFrame(import_frame)
        self.import_preview.pack(pady=5)
        
        # Initialize prediction manager
        self.prediction_manager = PredictionManager(self, None, self.controller, self.dataset)
        self._check_model_loaded()
        
    def update_probabilities(self, probabilities):
        """Update probability bars with new predictions"""
        for i, prob in enumerate(probabilities[0]):
            self.prob_bars[i].set(float(prob))
            self.prob_labels[i].configure(text=f"Digit {i}: {prob:.1%}")
            
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
            
    def _import_image(self):
        """Handle image file import"""
        try:
            filename = tk.filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg"),
                    ("All files", "*.*")
                ]
            )
            
            if not filename:
                return
                
            self._update_import_status("Loading image...", "orange")
            
            # Load and process image
            image = Image.open(filename).convert('L')  # Convert to grayscale
            processed_image = self._preprocess_imported_image(image)
            
            # Show preview and run prediction
            self.import_preview.display_image(processed_image)
            if self.is_model_loaded:
                asyncio.run(self.predict(processed_image))
                
            self._update_import_status("Image loaded successfully", "green")
            
        except Exception as e:
            self._update_import_status(f"Error: {str(e)}", "red")
            
    def _preprocess_imported_image(self, image):
        """Preprocess imported image for prediction"""
        # Resize to MNIST-like format (28x28)
        target_size = (28, 28)
        
        # Calculate dimensions to maintain aspect ratio
        width, height = image.size
        aspect = width / height
        
        if aspect > 1:
            new_width = int(target_size[0] * aspect)
            new_height = target_size[1]
        else:
            new_width = target_size[0]
            new_height = int(target_size[1] / aspect)
            
        # Resize and pad to square
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create square white background
        square = Image.new('L', (max(new_width, new_height),) * 2, 'white')
        
        # Paste resized image in center
        x = (square.width - resized.width) // 2
        y = (square.height - resized.height) // 2
        square.paste(resized, (x, y))
        
        # Final resize to target size
        final = square.resize(target_size, Image.LANCZOS)
        
        # Invert colors to match MNIST format
        return ImageOps.invert(final)
            
    def _import_mnist_csv(self):
        """Handle MNIST CSV file import"""
        try:
            # Open file dialog for CSV selection
            filename = tk.filedialog.askopenfilename(
                title="Select MNIST CSV File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not filename:
                return
                
            self._update_import_status("Loading CSV file...", "orange")
            
            # Load MNIST data
            self.dataset.load_mnist_csv(filename)
            
            if not self.dataset.processed_images:
                raise ValueError("No valid MNIST images found in CSV file")
                
            # Display first image as preview
            first_image = self.dataset.processed_images[0]
            self.import_preview.display_image(first_image)
            
            # Run prediction on imported image
            if self.is_model_loaded:
                asyncio.run(self.predict(first_image))
                
            self._update_import_status("CSV file loaded successfully", "green")
            
        except Exception as e:
            self._update_import_status(f"Error: {str(e)}", "red")
            
    def _update_import_status(self, message, color):
        """Update import status display"""
        self.import_status.configure(text=message, text_color=color)