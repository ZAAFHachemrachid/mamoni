import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import time
import os
import random
import numpy as np
import json

# Previeus codes
from Data_layer import ImageDataset
from Model_layer import NeuralNetwork # No need to import here, already defined in this file
from Visualization_Components import AnimatedHeatmap,TrainingMetrics

# Import the plotter classes
from plotNN_grok import NeuralNetworkPlotter, NetworkDataGenerator
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants (No changes needed)
FINAL_SIZE = 50
DEFAULT_CANVAS_SIZE = 400
DEFAULT_BRUSH_RADIUS = 15


class CanvasManager:
    """Manages the drawing canvas and its operations"""
    # ... (CanvasManager class code as you provided) ...
    def __init__(self, parent, canvas_size=DEFAULT_CANVAS_SIZE, brush_radius_var=None):
        self.parent = parent
        self.canvas_size = canvas_size
        self.brush_radius_var = brush_radius_var  # Store the IntVar directly
        self.stroke_history = []
        self.bbox_id = None
        self.bbox_coords = None

        # Create the canvas
        self.canvas = tk.Canvas( 
            parent, width=canvas_size, height=canvas_size,
            bg="white", bd=2, highlightthickness=2, highlightbackground="black"
        )

        # Create PIL image and drawing context
        self.image = Image.new("L", (canvas_size, canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Event bindings
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-3>", self.delete_stroke)

    def paint(self, event):
        """Draw on canvas and update internal image"""
        # Get brush radius directly from the IntVar
        r = self.brush_radius_var.get() if self.brush_radius_var else DEFAULT_BRUSH_RADIUS
        x, y = event.x, event.y
        oval_id = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.stroke_history.append((x - r, y - r, x + r, y + r, r, oval_id))
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")
        self.update_bbox()

    def delete_stroke(self, event):
        """Delete the closest stroke to the mouse pointer"""
        x, y = event.x, event.y
        if not self.stroke_history:
            return

        closest_stroke = min(
            self.stroke_history,
            key=lambda stroke: ((stroke[0] + stroke[2]) / 2 - x)**2 + ((stroke[1] + stroke[3]) / 2 - y)**2
        )

        self.canvas.delete(closest_stroke[-1])
        self.stroke_history.remove(closest_stroke)
        self.redraw_image()
        self.update_bbox()

    def update_bbox(self):
        """Update the bounding box around the drawing"""
        if self.bbox_id:
            self.canvas.delete(self.bbox_id)

        if self.stroke_history:
            x_coords = [stroke[0] for stroke in self.stroke_history] + [stroke[2] for stroke in self.stroke_history]
            y_coords = [stroke[1] for stroke in self.stroke_history] + [stroke[3] for stroke in self.stroke_history]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(self.canvas_size, x_max)
            y_max = min(self.canvas_size, y_max)
            self.bbox_coords = (x_min, y_min, x_max, y_max)
            self.bbox_id = self.canvas.create_rectangle(
                x_min, y_min, x_max, y_max,
                outline="red", width=2
            )
        else:
            self.bbox_coords = None
            self.bbox_id = None

    def undo(self):
        """Undo the last stroke"""
        if self.stroke_history:
            x1, y1, x2, y2, _, oval_id = self.stroke_history.pop()
            self.canvas.delete(oval_id)
            self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
            self.draw = ImageDraw.Draw(self.image)
            for stroke in self.stroke_history:
                x1, y1, x2, y2, _, _ = stroke
                self.draw.ellipse([x1, y1, x2, y2], fill="black")
            self.update_bbox()

    def redraw_image(self):
        """Redraw the entire image from stroke history"""
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        for x1, y1, x2, y2, radius, _ in self.stroke_history:
            self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def reset(self):
        """Reset canvas to blank state"""
        self.canvas.delete("all")
        self.stroke_history = []
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.bbox_id = None
        self.bbox_coords = None

    def resize(self, new_size):
        """Resize the canvas"""
        self.canvas_size = new_size
        self.canvas.config(width=new_size, height=new_size)
        self.reset()

    def get_processed_image(self):
        """Return processed image ready for saving or prediction"""
        if self.bbox_coords:
            x_min, y_min, x_max, y_max = self.bbox_coords
            cropped_image = self.image.crop((x_min, y_min, x_max, y_max))
            image_to_process = cropped_image
        else:
            image_to_process = self.image

        # Resize and invert for MNIST-like format
        return ImageOps.invert(image_to_process.resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS))

class ImagePreviewFrame:
    """Manages the preview of saved images"""
    # ... (ImagePreviewFrame class code as you provided) ...
    def __init__(self, parent):
        self.parent = parent
        self.preview_container = ttk.LabelFrame(parent, text="Last Saved Image")
        self.preview_label = tk.Label(self.preview_container, bd=2, relief="groove")
        self.photo_image = None

        # Layout
        self.preview_label.pack(padx=10, pady=10)

        # Initialize with blank image
        self.display_image()

    def display_image(self, image=None):
        """Display an image in the preview panel"""
        if image is None:
            img = Image.new("L", (FINAL_SIZE, FINAL_SIZE), "white")
        else:
            img = image

        # Ensure image is correctly sized
        img_resized = img.resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(img_resized)
        self.preview_label.config(image=self.photo_image)

    def pack(self, **kwargs):
        """Pack the preview container with given options"""
        self.preview_container.pack(**kwargs)

class PredictionManager:
    """Handles digit prediction functionality"""
    # ... (PredictionManager class code as you provided) ...
    def __init__(self, parent, prediction_label, controller, dataset): # ADD controller and dataset
        self.parent = parent
        self.prediction_label = prediction_label
        self.controller = controller # ADDED
        self.dataset = dataset # ADDED

    def predict(self, image):
        """Predict digit from image using the trained model"""
        if self.controller.model is None:
            messagebox.showerror("Error", "Model not trained or loaded yet. Please go to 'Neural Network Trainer' tab and train or load a model.")
            self.prediction_label.config(text="Prediction: Model not ready") # Update label even if error
            return None

        try:
            # 1. Preprocess the image to get features (same as training)
            feature_method = self.controller.dataset.feature_method # Corrected attribute name here
            feature_size = self.controller.dataset.current_feature_size # Get from controller dataset

            processed_image_np = np.array(image) / 255.0 # Normalize like in dataset loading

            features = self.dataset.image_to_features( # Use dataset instance to extract features
                processed_image_np,
                method=feature_method,
                feature_size=feature_size
            )
            features = features.reshape(1, -1) # Reshape to (1, num_features) for single prediction

            # 2. Use the model to predict
            prediction = self.controller.model.predict(features)[0] # Get single prediction

            # 3. Update the prediction label
            self.prediction_label.config(text=f"Prediction: {prediction}")
            return prediction

        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {e}")
            self.prediction_label.config(text="Prediction: Error") # Update label in case of error
            return None


class FileManager:
    """Manages file operations and naming"""
    # ... (FileManager class code as you provided) ...
    def __init__(self, filename_var, auto_count, start_number):
        self.filename_var = filename_var
        self.auto_count = auto_count
        self.start_number = start_number
        self.last_saved_path = None

        # Create dataset directory if it doesn't exist
        os.makedirs("dataset", exist_ok=True)

    def save_image(self, image):
        """Save the processed image to file"""
        filename = self.filename_var.get()

        # Create filename based on settings
        if self.auto_count.get():
            file_id = f"{filename}_{self.start_number.get()}"
            # Increment counter after save
            self.start_number.set(self.start_number.get() + 1)
        else:
            file_id = f"{filename}_{int(time.time())}"

        # Complete path
        file_path = os.path.join("dataset", f"{file_id}.png")

        # Save the image
        image.save(file_path)
        self.last_saved_path = file_path

        print(f"Image saved as {file_path}")
        return file_path

class ControlPanel:
    """Manages all control widgets and settings"""

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.plotter = None  # Initialize plotter here

        # Create frames for organized controls
        self.create_brush_settings()
        self.create_canvas_settings()
        self.create_file_management()
        self.create_auto_counter()
        self.create_action_buttons()

        # Create preview frame
        self.preview = ImagePreviewFrame(parent)
        self.preview.pack(fill="x", pady=5)

    def create_brush_settings(self):
        """Create brush settings controls"""
        brush_frame = ttk.LabelFrame(self.parent, text="Brush Settings")
        brush_frame.pack(fill="x", pady=5)

        tk.Scale(
            brush_frame, from_=1, to=20, orient=tk.HORIZONTAL,
            variable=self.app.brush_radius, label="Brush Radius:"
        ).pack(fill="x")

    def create_canvas_settings(self):
        """Create canvas settings controls"""
        canvas_frame = ttk.LabelFrame(self.parent, text="Canvas Settings")
        canvas_frame.pack(fill="x", pady=5)

        tk.Scale(
            canvas_frame, from_=200, to=600, orient=tk.HORIZONTAL,
            variable=self.app.canvas_size, label="Canvas Size:",
            command=self.app.resize_canvas
        ).pack(fill="x")

    def create_file_management(self):
        """Create file management controls"""
        file_frame = ttk.LabelFrame(self.parent, text="File Management")
        file_frame.pack(fill="x", pady=5)

        # Radio buttons for digits 0-9
        for i in range(10):
            digit_str = str(i)
            tk.Radiobutton(
                file_frame,
                text=digit_str,
                variable=self.app.filename_var,
                value=digit_str,
                indicatoron=0,
                width=3
            ).pack(side="left", padx=2, pady=2)

        # Set default to "unknown"
        self.app.filename_var.set("unknown")

    def create_auto_counter(self):
        """Create auto counter controls"""
        auto_frame = ttk.LabelFrame(self.parent, text="Auto Counting")
        auto_frame.pack(fill="x", pady=5)

        tk.Checkbutton(
            auto_frame, text="Enable Auto-count", variable=self.app.auto_count,
            command=self.toggle_auto_count
        ).pack(side="left", padx=5)

        self.spinbox = tk.Spinbox(
            auto_frame, from_=1, to=1000, textvariable=self.app.start_number,
            state="disabled", width=8
        )
        self.spinbox.pack(side="right", padx=5)

    def toggle_auto_count(self):
        """Toggle auto count spinbox state"""
        self.spinbox.config(state='normal' if self.app.auto_count.get() else 'disabled')

    def create_action_buttons(self):
        """Create action buttons"""
        action_frame = ttk.LabelFrame(self.parent, text="Actions")
        action_frame.pack(fill="x", pady=5)

        # Configure grid layout
        for i in range(5): # Increased to 5 columns to include "Model Preview"
            action_frame.grid_columnconfigure(i, weight=1)

        # Create buttons
        tk.Button(action_frame, text="Undo (d)", command=self.app.undo).grid(
            row=0, column=0, sticky="ew", padx=2, pady=2
        )
        tk.Button(action_frame, text="Save (s)", command=self.app.save_image).grid(
            row=0, column=1, sticky="ew", padx=2, pady=2
        )
        tk.Button(action_frame, text="Reset (r)", command=self.app.reset_canvas).grid(
            row=0, column=2, sticky="ew", padx=2, pady=2
        )
        tk.Button(action_frame, text="Predict", command=self.app.predict_digit).grid(
            row=0, column=3, sticky="ew", padx=2, pady=2
        )
        tk.Button(action_frame, text="Model Preview", command=self.app.open_model_preview_popup).grid( # Added button
            row=0, column=4, sticky="ew", padx=2, pady=2
        )


        # Create prediction label
        self.prediction_label = tk.Label(action_frame, text="Prediction: ")
        self.prediction_label.grid(row=1, column=0, columnspan=5, sticky="ew", padx=2, pady=5) # Updated columnspan

        # Set prediction label
        self.app.prediction_label = self.prediction_label


class NeuralNetController:
    """Handles training and evaluation of neural network models."""
    # ... (NeuralNetController class code - No changes needed) ...
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.model = None

    def set_dataset(self, dataset):
        """Set the dataset to use for training/testing."""
        self.dataset = dataset

    def create_model(self, layer_sizes=None, input_size=None): # Added input_size
        """Create a new neural network model with specified architecture."""
        if layer_sizes is None:
            if input_size is None: # Use dataset feature size if input_size not provided
                if self.dataset and self.dataset.features is not None:
                    input_size = self.dataset.features.shape[1]
                else:
                    raise ValueError("Cannot create model: no dataset loaded or layer_sizes specified")
            output_size = self.dataset.num_classes
            layer_sizes = [input_size] + layer_sizes + [output_size] # Fix: Use provided layer_sizes as hidden layers
        self.model = NeuralNetwork(layer_sizes) # Assuming NeuralNetwork is defined in Model_layer.py
        return self.model

    def train_epoch(self, learning_rate, batch_size):
        """Train the model for one epoch and return metrics."""
        if self.model is None:
            raise ValueError("No model initialized. Call create_model first.")

        if self.dataset is None or self.dataset.train_features is None: # Use train_features
            raise ValueError("No dataset loaded or features not prepared and split.")

        loss = self.model.train_epoch(
            self.dataset.train_features, # Use train_features
            self.dataset.train_labels, # Use train_labels
            lr=learning_rate,
            batch_size=batch_size
        )

        predictions = self.model.predict(self.dataset.train_features) # Use train_features for train accuracy
        accuracy = np.mean(predictions == np.argmax(self.dataset.train_labels, axis=1)) # Use train_labels for train accuracy

        return loss, accuracy

    def validate(self): # New validate method
        """Validate current model on validation dataset."""
        if self.model is None:
            raise ValueError("No model initialized. Call create_model first.")

        if self.dataset is None or self.dataset.val_features is None: # Use val_features
            raise ValueError("No dataset loaded or features not prepared and split.")

        predictions = self.model.predict(self.dataset.val_features) # Use val_features
        accuracy = np.mean(predictions == np.argmax(self.dataset.val_labels, axis=1)) # Use val_labels

        # Calculate cross-entropy loss on validation set
        activations = self.model.forward(self.dataset.val_features)
        log_probs = np.log(activations[-1] + 1e-8)
        val_loss = -np.sum(self.dataset.val_labels.reshape(self.dataset.val_labels.shape[0], -1) * log_probs) / self.dataset.val_labels.shape[0]


        return val_loss, accuracy


    def evaluate(self):
        """Evaluate current model on test dataset."""
        if self.model is None:
            raise ValueError("No model initialized. Call create_model first.")

        if self.dataset is None or self.dataset.test_features is None: # Use test_features
            raise ValueError("No dataset loaded or features not prepared and split.")

        predictions = self.model.predict(self.dataset.test_features) # Use test_features
        accuracy = np.mean(predictions == np.argmax(self.dataset.test_labels, axis=1)) # Use test_labels

        return accuracy

    def save_model(self, filepath):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        self.model.save_model(filepath)

    def load_model(self, filepath, input_size=None): # Added input_size
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

class ProgressManager:
    """Manages progress bar and status updates."""
    # ... (ProgressManager class code - No changes needed) ...
    def __init__(self, progress_bar, progress_label, root):
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        self.root = root

    def start(self, mode='indeterminate', text="Processing..."):
        """Start progress indication."""
        self.progress_bar['mode'] = mode
        if mode == 'indeterminate':
            self.progress_bar.start()
        self.progress_label.config(text=text)
        self.root.update()

    def update(self, value, max_value, text=None):
        """Update progress bar with current value."""
        self.progress_bar['mode'] = 'determinate'
        self.progress_bar['maximum'] = max_value
        self.progress_bar['value'] = value
        if text:
            self.progress_label.config(text=text)
        self.root.update_idletasks()

    def stop(self, text="Done"):
        """Stop progress indication."""
        self.progress_bar.stop()
        self.progress_label.config(text=text)
        self.progress_bar['mode'] = 'determinate'
        self.root.update()

class NeuralNetworkAppTab(ttk.Frame): # Inherit from ttk.Frame to be a Tab
    """Neural Network Training Application as a Tab"""

    def __init__(self, parent):
        super().__init__(parent) # Initialize ttk.Frame
        self.root = parent # Keep parent for progress manager, but it's a Frame now, not Tk()
        self.dataset = ImageDataset() # Assuming ImageDataset is defined in Data_layer.py
        self.controller = NeuralNetController(self.dataset)
        self.feature_method_var = tk.StringVar(value='average') # Default feature method
        self.feature_size_var = tk.StringVar(value='5x5') # Default feature size
        self.hidden_layers_var = tk.StringVar(value='64') # Default hidden layers
        # Dataset split ratios
        self.train_ratio_var = tk.DoubleVar(value=0.7) # Default train ratio
        self.val_ratio_var = tk.DoubleVar(value=0.15) # Default validation ratio
        self.test_ratio_var = tk.DoubleVar(value=0.15) # Default test ratio

        self.dataset_path_var = tk.StringVar(value=r"D:\python\tp_nn_final_gui\MyData")
        self.images_per_class_var = tk.StringVar(value="10")

        self.epochs_var = tk.StringVar(value="20")
        self.learning_rate_var = tk.StringVar(value="0.1")
        self.batch_size_var = tk.StringVar(value="128")

        # UI components
        self._init_ui()

    def _init_ui(self):
        """Initialize all UI components with new frame layout."""
        # Data Frame
        self._create_data_frame(self)
        self.data_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Train Frame
        self._create_train_frame(self)
        self.train_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Progress Frame
        self._create_progress_frame()
        self.progress_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        # Configure grid weights for resizing
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def _create_data_frame(self, parent):
        """Creates the Data Processing Frame."""
        self.data_frame = ttk.Frame(parent, padding=10) # Parent is now passed argument

        # Heatmap Frame (inside Data Frame)
        heatmap_params = {'data_shape': (50, 50), 'cmap': 'gray', 'vmin': 0, 'vmax': 255, 'figsize': (3, 3)} # Smaller heatmap
        self.heatmap = AnimatedHeatmap(self.data_frame, params=heatmap_params) # Assuming AnimatedHeatmap is defined in Visualization_Components.py
        self.heatmap.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Dataset Path Frame
        dataset_path_frame = ttk.Frame(self.data_frame)
        dataset_path_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(dataset_path_frame, text="Dataset Path:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.dataset_path_var = tk.StringVar(value=r"D:\python\tp_nn_final_gui\MyData") # Example path
        dataset_path_entry = ttk.Entry(dataset_path_frame, textvariable=self.dataset_path_var, width=30)
        dataset_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        dataset_path_button = ttk.Button(dataset_path_frame, text="Browse", command=self._browse_dataset_path)
        dataset_path_button.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        dataset_path_button.bind("<Enter>", lambda event: self._show_tooltip(dataset_path_button, "Browse your dataset directory"))
        dataset_path_entry.bind("<Enter>", lambda event: self._show_tooltip(dataset_path_entry, "Path to the dataset directory"))

        # Images Per Class Frame
        images_per_class_frame = ttk.Frame(self.data_frame)
        images_per_class_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(images_per_class_frame, text="Images/Class:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.images_per_class_var = tk.StringVar(value="10")
        images_per_class_entry = ttk.Entry(images_per_class_frame, textvariable=self.images_per_class_var, width=5)
        images_per_class_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        images_per_class_entry.bind("<Enter>", lambda event: self._show_tooltip(images_per_class_entry, "Max images to load per class"))

        # Feature Method Frame
        self._create_feature_method_frame(self.data_frame, row_num=3) # Pass data_frame and row_num
        # Feature Size Frame
        self._create_feature_size_frame(self.data_frame, row_num=4) # Pass data_frame and row_num

        # Data Buttons Frame
        data_buttons_frame = ttk.Frame(self.data_frame)
        data_buttons_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
        self.load_dataset_button = ttk.Button(data_buttons_frame, text="Load Dataset", command=self._load_dataset)
        self.load_dataset_button.grid(row=0, column=0, sticky="ew", padx=2, pady=5)
        self.prepare_data_button = ttk.Button(data_buttons_frame, text="Prepare Data", command=self._prepare_data)
        self.prepare_data_button.grid(row=0, column=1, sticky="ew", padx=2, pady=5)
        self.load_dataset_button.bind("<Enter>", lambda event: self._show_tooltip(self.load_dataset_button, "Load dataset from directory"))
        self.prepare_data_button.bind("<Enter>", lambda event: self._show_tooltip(self.prepare_data_button, "Prepare features from loaded dataset"))

        self.data_frame.grid_columnconfigure(0, weight=1) # For heatmap to expand
        self.data_frame.grid_columnconfigure(1, weight=1) # For other widgets to expand


    def _create_train_frame(self, parent):
        """Creates the Training Frame."""
        self.train_frame = ttk.Frame(parent, padding=10) # Parent is now passed argument

        # Metrics Plot Frame (inside Train Frame)
        self.metrics_plot = TrainingMetrics(self.train_frame) # Assuming TrainingMetrics is defined in Visualization_Components.py
        self.metrics_plot.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Hidden Layer Frame
        self._create_hidden_layer_frame(self.train_frame, row_num=1) # Pass train_frame and row_num
        # Dataset Split Frame
        self._create_dataset_split_frame(self.train_frame, row_num=2) # Pass train_frame and row_num
        # Training Parameters Frame
        self._create_training_params_frame(self.train_frame, row_num=3) # Pass train_frame and row_num

        # Training Buttons Frame
        train_buttons_frame = ttk.Frame(self.train_frame)
        train_buttons_frame.grid(row=4, column=0, sticky="ew", pady=5)
        self.creat_model_button = ttk.Button(train_buttons_frame, text="Create Model", command=self._creat_model)
        self.creat_model_button.grid(row=0, column=0, sticky="ew", padx=2, pady=5)
        self.train_model_button = ttk.Button(train_buttons_frame, text="Train Model", command=self._train_model)
        self.train_model_button.grid(row=0, column=1, sticky="ew", padx=2, pady=5)
        self.test_model_button = ttk.Button(train_buttons_frame, text="Test Model", command=self._test_model)
        self.test_model_button.grid(row=0, column=2, sticky="ew", padx=2, pady=5)
        self.creat_model_button.bind("<Enter>", lambda event: self._show_tooltip(self.creat_model_button, "Create a new model"))
        self.train_model_button.bind("<Enter>", lambda event: self._show_tooltip(self.train_model_button, "Train the model"))
        self.test_model_button.bind("<Enter>", lambda event: self._show_tooltip(self.test_model_button, "Test the trained model"))

        # Model I/O Buttons Frame
        model_io_frame = ttk.Frame(self.train_frame)
        model_io_frame.grid(row=5, column=0, sticky="ew", pady=5)
        self.save_model_button = ttk.Button(model_io_frame, text="Save Model", command=self._save_model)
        self.save_model_button.grid(row=0, column=0, sticky="ew", padx=2)
        self.load_model_button = ttk.Button(model_io_frame, text="Load Model", command=self._load_model)
        self.load_model_button.grid(row=0, column=1, sticky="ew", padx=2)
        self.export_features_button = ttk.Button(model_io_frame, text="Export Features", command=self._export_features)
        self.export_features_button.grid(row=0, column=2, sticky="ew", padx=2)
        self.save_model_button.bind("<Enter>", lambda event: self._show_tooltip(self.save_model_button, "Save the current model"))
        self.load_model_button.bind("<Enter>", lambda event: self._show_tooltip(self.load_model_button, "Load a model from file"))
        self.export_features_button.bind("<Enter>", lambda event: self._show_tooltip(self.export_features_button, "Export prepared features to CSV"))

        self.train_frame.grid_columnconfigure(0, weight=1) # For plot and other content to expand


    def _create_progress_frame(self):
        """Creates the Progress Bar Frame."""
        self.progress_frame = ttk.Frame(self, padding=10) # Parent is now 'self' (tab frame)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.progress_label = ttk.Label(self.progress_frame, text="")
        self.progress_label.pack(pady=5)
        self.progress_manager = ProgressManager(self.progress_bar, self.progress_label, self.root) # Use self.root (main window)


    def _create_feature_method_frame(self, parent_frame, row_num):
        """Creates the feature method selection frame."""
        feature_method_frame = ttk.LabelFrame(parent_frame, text="Feature Method", padding=5)
        feature_method_frame.grid(row=row_num, column=0, columnspan=2, sticky="ew", pady=5)

        methods = ['average', 'sum', 'max']
        for i, method in enumerate(methods):
            rb = ttk.Radiobutton(feature_method_frame, text=method.capitalize(), variable=self.feature_method_var, value=method)
            rb.grid(row=0, column=i, padx=10, pady=5, sticky='w')
            rb.bind("<Enter>", lambda event: self._show_tooltip(rb, f"Select {method.capitalize()} pooling"))

    def _create_feature_size_frame(self, parent_frame, row_num):
        """Creates the feature size selection frame."""
        feature_size_frame = ttk.LabelFrame(parent_frame, text="Feature Size", padding=5)
        feature_size_frame.grid(row=row_num, column=0, columnspan=2, sticky="ew", pady=5)

        sizes = ['5x5', '10x10', '25x25', '50x50 (No Prep)']
        for i, size in enumerate(sizes):
            rb = ttk.Radiobutton(feature_size_frame, text=size, variable=self.feature_size_var, value=size)
            rb.grid(row=0, column=i, padx=10, pady=5, sticky='w')
            rb.bind("<Enter>", lambda event: self._show_tooltip(rb, f"Select feature size: {size}"))

    def _create_hidden_layer_frame(self, parent_frame, row_num):
        """Creates the hidden layer size input frame."""
        hidden_layer_frame = ttk.LabelFrame(parent_frame, text="Hidden Layers", padding=5)
        hidden_layer_frame.grid(row=row_num, column=0, sticky="ew", pady=5)

        ttk.Label(hidden_layer_frame, text="Sizes (comma-separated):").grid(row=0, column=0, sticky=tk.W, padx=5)
        hidden_layers_entry = ttk.Entry(hidden_layer_frame, textvariable=self.hidden_layers_var, width=30)
        hidden_layers_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        hidden_layer_frame.grid_columnconfigure(1, weight=1)
        hidden_layers_entry.bind("<Enter>", lambda event: self._show_tooltip(hidden_layers_entry, "Hidden layer sizes, e.g., '64,32'"))

    def _create_dataset_split_frame(self, parent_frame, row_num):
        """Creates the dataset split ratio frame with Entry widgets."""
        split_frame = ttk.LabelFrame(parent_frame, text="Dataset Split Ratios", padding=5)
        split_frame.grid(row=row_num, column=0, sticky="ew", pady=5)

        # Train ratio input
        ttk.Label(split_frame, text="Train Ratio:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.train_ratio_entry = ttk.Entry(split_frame, textvariable=self.train_ratio_var, width=5)
        self.train_ratio_entry.grid(row=0, column=1, sticky="ew", padx=5)
        self.train_ratio_entry.bind("<FocusOut>", self._validate_and_normalize_ratios)
        self.train_ratio_entry.bind("<Enter>", lambda event: self._show_tooltip(self.train_ratio_entry, "Train dataset ratio (0.0-1.0)"))

        # Validation ratio input
        ttk.Label(split_frame, text="Validation Ratio:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.val_ratio_entry = ttk.Entry(split_frame, textvariable=self.val_ratio_var, width=5)
        self.val_ratio_entry.grid(row=1, column=1, sticky="ew", padx=5)
        self.val_ratio_entry.bind("<FocusOut>", self._validate_and_normalize_ratios)
        self.val_ratio_entry.bind("<Enter>", lambda event: self._show_tooltip(self.val_ratio_entry, "Validation dataset ratio (0.0-1.0)"))

        # Test ratio input
        ttk.Label(split_frame, text="Test Ratio:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.test_ratio_entry = ttk.Entry(split_frame, textvariable=self.test_ratio_var, width=5)
        self.test_ratio_entry.grid(row=2, column=1, sticky="ew", padx=5)
        self.test_ratio_entry.bind("<FocusOut>", self._validate_and_normalize_ratios)
        self.test_ratio_entry.bind("<Enter>", lambda event: self._show_tooltip(self.test_ratio_entry, "Test dataset ratio (0.0-1.0)"))

        split_frame.grid_columnconfigure(1, weight=1)
        self._validate_and_normalize_ratios()

    def _create_training_params_frame(self, parent_frame, row_num):
        """Create the training parameters section."""
        training_params_frame = ttk.LabelFrame(parent_frame, text="Training Parameters", padding=5)
        training_params_frame.grid(row=row_num, column=0, sticky="ew", pady=5)

        # Epochs
        ttk.Label(training_params_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, padx=5)
        epochs_entry = ttk.Entry(training_params_frame, textvariable=self.epochs_var, width=5)
        epochs_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        epochs_entry.bind("<Enter>", lambda event: self._show_tooltip(epochs_entry, "Number of epochs"))

        # Learning Rate
        ttk.Label(training_params_frame, text="Learning Rate:").grid(row=0, column=2, sticky=tk.W, padx=5)
        lr_entry = ttk.Entry(training_params_frame, textvariable=self.learning_rate_var, width=5)
        lr_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        lr_entry.bind("<Enter>", lambda event: self._show_tooltip(lr_entry, "Learning rate"))

        # Batch Size
        ttk.Label(training_params_frame, text="Batch Size:").grid(row=0, column=4, sticky=tk.W, padx=5)
        batch_size_entry = ttk.Entry(training_params_frame, textvariable=self.batch_size_var, width=5)
        batch_size_entry.grid(row=0, column=5, sticky=tk.W, padx=5)
        batch_size_entry.bind("<Enter>", lambda event: self._show_tooltip(batch_size_entry, "Batch size"))

    def _validate_and_normalize_ratios(self, event=None):
        """Validates ratio entries and normalizes them to sum to 1.0."""
        try:
            train_ratio = float(self.train_ratio_var.get())
            val_ratio = float(self.val_ratio_var.get())
            test_ratio = float(self.test_ratio_var.get())

            if train_ratio < 0 or train_ratio > 1 or val_ratio < 0 or val_ratio > 1 or test_ratio < 0 or test_ratio > 1:
                messagebox.showerror("Error", "Ratios must be between 0.0 and 1.0.")
                return

            total_ratio = train_ratio + val_ratio + test_ratio

            if not np.isclose(total_ratio, 1.0):
                if total_ratio > 0:
                    train_ratio = train_ratio / total_ratio
                    val_ratio = val_ratio / total_ratio
                    test_ratio = test_ratio / total_ratio
                else:
                    train_ratio = 0.7
                    val_ratio = 0.15
                    test_ratio = 0.15

                self.train_ratio_var.set(f"{train_ratio:.2f}")
                self.val_ratio_var.set(f"{val_ratio:.2f}")
                self.test_ratio_var.set(f"{test_ratio:.2f}")


        except ValueError:
            messagebox.showerror("Error", "Invalid ratio value. Please enter numbers between 0.0 and 1.0.")

    def _show_tooltip(self, widget, text):
        """Display tooltip for the given widget."""
        tooltip = ToolTip(widget, text=text) # Assuming ToolTip is defined in Script 2

    def _browse_dataset_path(self):
        """Browse for dataset directory."""
        root_dir = filedialog.askdirectory(title="Select Dataset Directory")
        if root_dir:
            self.dataset_path_var.set(root_dir)

    def _load_dataset(self):
        """Load image dataset from selected directory."""
        root_dir = self.dataset_path_var.get()
        if not root_dir:
            messagebox.showerror("Error", "Dataset path is required.")
            return

        try:
            max_num_img_per_label = int(self.images_per_class_var.get())
            if max_num_img_per_label <= 0:
                messagebox.showerror("Error", "Images per class must be positive.")
                return

            self.progress_manager.start(mode='indeterminate', text="Loading Dataset...")
            self.dataset.load_dataset(
                root_dir,
                max_num_img_per_label,
                progress_callback=self.progress_manager.update,
                heatmap_callback=self.heatmap.update_heatmap
            )
            self.progress_manager.stop(text="Dataset Loaded")
            messagebox.showinfo("Info", "Dataset loaded and cropped successfully!")

        except ValueError:
            self.progress_manager.stop(text="Load Failed")
            messagebox.showerror("Error", "Invalid Images per Class value.")
        except FileNotFoundError:
            self.progress_manager.stop(text="Load Failed")
            messagebox.showerror("Error", "Dataset directory not found.")
        except Exception as e:
            self.progress_manager.stop(text="Load Failed")
            messagebox.showerror("Error", f"Dataset loading error: {e}")

    def _prepare_data(self):
        """Prepare features from the loaded dataset and split dataset."""
        if not self.dataset.processed_images:
            messagebox.showerror("Error", "Dataset must be loaded first.")
            return

        try:
            self.progress_manager.start(mode='indeterminate', text="Preparing Data...")
            feature_method = self.feature_method_var.get() # Get selected method
            feature_size_str = self.feature_size_var.get() # Get selected feature size string
            if feature_size_str == '5x5':
                feature_size = (5, 5)
            elif feature_size_str == '10x10':
                feature_size = (10, 10)
            elif feature_size_str == '25x25':
                feature_size = (25, 25)
            elif feature_size_str == '50x50 (No Prep)':
                feature_size = (50, 50)
            else:
                feature_size = (5, 5) # Default case

            self.dataset.prepare_features(
                progress_callback=self.progress_manager.update,
                heatmap_callback=self.heatmap.update_heatmap,
                feature_method=feature_method, # Pass method
                feature_size=feature_size # Pass feature size
            )

            # Split dataset after preparing features
            train_ratio = self.train_ratio_var.get()
            val_ratio = self.val_ratio_var.get()
            test_ratio = self.test_ratio_var.get()
            self.dataset.split_dataset(train_ratio, val_ratio, test_ratio)


            self.progress_manager.stop(text="Data Prepared and Split")
            messagebox.showinfo("Info", f"Dataset features prepared successfully using {feature_method.capitalize()} method and feature size {feature_size_str}, and dataset split!")

        except ValueError as e:
            self.progress_manager.stop(text="Preparation Failed")
            messagebox.showerror("Error", str(e))
        except Exception as e:
            self.progress_manager.stop(text="Preparation Failed")
            messagebox.showerror("Error", f"Data preparation error: {e}")

    def _creat_model(self):
        """create model after data prepared to know input size"""
        if self.dataset.features is None:
            messagebox.showerror("Error", "Dataset features not prepared. Prepare data first.")
            return

        try:
            self.progress_manager.start(mode='indeterminate', text="Creat Model...")

            # Create model after data is prepared to know input size
            hidden_layers_str = self.hidden_layers_var.get()
            hidden_layer_sizes = [int(size) for size in hidden_layers_str.split(',') if size.strip()] # Parse hidden layer sizes
            feature_size = self.dataset.current_feature_size
            input_size = np.prod(feature_size) if feature_size != (50,50) else 50*50 # Calculate input size dynamically
            layer_sizes = [input_size] + hidden_layer_sizes + [self.dataset.num_classes] # Construct layer sizes
            self.controller.create_model(layer_sizes=layer_sizes) # Create model with dynamic layer sizes

            self.progress_manager.stop(text="Model Created")
            messagebox.showinfo("Info", f"Model Created successfully")

        except ValueError as e:
            self.progress_manager.stop(text="Creat Model Failed")
            messagebox.showerror("Error", str(e))
        except Exception as e:
            self.progress_manager.stop(text="Creat Model Failed")
            messagebox.showerror("Error", f"Model creating error: {e}")

    def _train_model(self):
        """Train the neural network model."""
        if self.controller.model is None:
            messagebox.showerror("Error", "Model not initialized. Prepare data first.")
            return
        if self.dataset.train_features is None: # Use train_features
            messagebox.showerror("Error", "Dataset features not prepared and split. Prepare data first.")
            return

        try:
            epochs = int(self.epochs_var.get())
            lr = float(self.learning_rate_var.get())
            batch_size = int(self.batch_size_var.get())

            if epochs <= 0 or lr <= 0 or batch_size <= 0:
                messagebox.showerror("Error", "Training parameters must be positive.")
                return

            self.metrics_plot.reset()
            self.progress_manager.start(mode='determinate', text="Training Model...")

            for epoch in range(epochs):
                loss, accuracy = self.controller.train_epoch(lr, batch_size)
                val_loss, val_accuracy = self.controller.validate() # Get validation metrics
                self.metrics_plot.update_plot(epoch + 1, loss, accuracy, val_loss, val_accuracy) # Update plot with validation metrics
                self.progress_manager.update(epoch + 1, epochs,
                                             text=f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                self.root.update()
                time.sleep(0.01) # Small delay to visualize progress

            self.progress_manager.stop(text="Training Finished")
            messagebox.showinfo("Info", "Model training completed!")

        except ValueError:
            self.progress_manager.stop(text="Training Failed")
            messagebox.showerror("Error", "Invalid training parameter value.")
        except Exception as e:
            self.progress_manager.stop(text="Training Failed")
            messagebox.showerror("Error", f"Model training error: {e}")

    def _test_model(self):
        """Test the trained neural network model."""
        if self.controller.model is None:
            messagebox.showerror("Error", "Model not trained or loaded yet.")
            return
        if self.dataset.test_features is None: # Use test_features
            messagebox.showerror("Error", "Dataset features not prepared and split. Prepare data first.")
            return

        try:
            self.progress_manager.start(mode='indeterminate', text="Testing Model...")
            accuracy = self.controller.evaluate()
            self.progress_manager.stop(text=f"Testing Finished. Accuracy: {accuracy:.4f}")
            messagebox.showinfo("Info", f"Model testing completed! Accuracy: {accuracy:.4f}")

        except Exception as e:
            self.progress_manager.stop(text="Testing Failed")
            messagebox.showerror("Error", f"Model testing error: {e}")

    def _save_model(self):
        """Save the current model to a file."""
        if self.controller.model is None:
            messagebox.showerror("Error", "No model to save. Train or load a model first.")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if filepath:
            try:
                self.progress_manager.start(mode='indeterminate', text="Saving Model...")
                self.controller.save_model(filepath)
                self.progress_manager.stop(text="Model Saved")
                messagebox.showinfo("Info", "Model saved successfully!")
            except Exception as e:
                self.progress_manager.stop(text="Save Failed")
                messagebox.showerror("Error", f"Error saving model: {e}")
                print(f"Error saving model: {e}")

    def _load_model(self):
        """Load a model from a file."""
        filepath = filedialog.askopenfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if filepath:
            try:
                self.progress_manager.start(mode='indeterminate', text="Loading Model...")
                feature_size = self.dataset.current_feature_size # Get current feature size for input layer size
                input_size = np.prod(feature_size) if feature_size != (50,50) else 50*50 # Calculate input size
                self.controller.load_model(filepath, input_size=input_size) # Pass input_size to load_model
                self.progress_manager.stop(text="Model Loaded")
                messagebox.showinfo("Info", "Model loaded successfully!")
            except Exception as e:
                self.progress_manager.stop(text="Load Failed")
                messagebox.showerror("Error", f"Error loading model: {e}")

    def _export_features(self):
        """Export prepared features to a CSV file."""
        if self.dataset.features is None:
            messagebox.showerror("Error", "No features prepared to export.")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filepath:
            try:
                self.progress_manager.start(mode='indeterminate', text="Exporting Features...")
                self.dataset.export_to_csv(filepath)
                self.progress_manager.stop(text="Features Exported")
                messagebox.showinfo("Info", "Features exported to CSV successfully!")
            except ValueError as e:
                self.progress_manager.stop(text="Export Failed")
                messagebox.showerror("Error", str(e))
            except Exception as e:
                self.progress_manager.stop(text="Export Failed")
                messagebox.showerror("Error", f"Error exporting features: {e}")

class ToolTip: # ToolTip Class from Script 2
    """Tooltip class for providing help text on hover."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tooltip_window = tk.Toplevel(self.widget)
        tooltip_window.wm_overrideredirect(True) # removes window border
        tooltip_window.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(tooltip_window, text=self.text, background="#f9f9f9", relief=tk.SOLID, borderwidth=1, padding=5)
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

class MergedApp(tk.Tk): # Inherit from tk.Tk

    def __init__(self):
        super().__init__() # Initialize tk.Tk
        self.title("Merged Application")

        # Configure main grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")


        # Create Neural Network App Tab
        self.nn_tab_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.nn_tab_frame, text='Neural Network Trainer')
        self.nn_app_tab = NeuralNetworkAppTab(self.nn_tab_frame)
        self.nn_tab_frame.grid_columnconfigure(0, weight=1) # Configure grid for nn_tab_frame
        self.nn_tab_frame.grid_rowconfigure(0, weight=1)    # Configure grid for nn_tab_frame
        # Add this line to place nn_app_tab inside nn_tab_frame
        self.nn_app_tab.grid(row=0, column=0, sticky="nsew") # Place NeuralNetworkAppTab in nn_tab_frame


        # Create Drawing App Tab
        self.drawing_tab_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.drawing_tab_frame, text='Drawing App')
        self.drawing_app_tab = DrawingAppTab(self.drawing_tab_frame, self.nn_app_tab.controller, self.nn_app_tab.dataset) # MODIFIED - Pass controller and dataset
        self.drawing_tab_frame.grid_columnconfigure(0, weight=1)
        self.drawing_tab_frame.grid_rowconfigure(0, weight=1)


        # Setup keyboard shortcuts for Drawing Tab (assuming Drawing Tab is initially selected)
        for char, command in {'s': self.drawing_app_tab.save_image, 'r': self.drawing_app_tab.reset_canvas, 'd': self.drawing_app_tab.undo}.items():
            self.bind(f'<{char}>', lambda event, cmd=command: cmd())
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def on_tab_changed(self, event):
        """Handle tab change event to update keyboard shortcuts."""
        selected_tab = self.notebook.select()
        tab_text = self.notebook.tab(selected_tab, "text")

        # Unbind existing shortcuts
        for char in ['s', 'r', 'd']:
            self.unbind(f'<{char}>')

        if tab_text == 'Drawing App':
            for char, command in {'s': self.drawing_app_tab.save_image, 'r': self.drawing_app_tab.reset_canvas, 'd': self.drawing_app_tab.undo}.items():
                self.bind(f'<{char}>', lambda event, cmd=command: cmd())
        # Add shortcuts for other tabs if needed in the future




# ======================================
# Model Layer (Already included in your provided code)
# ======================================
class NeuralNetwork:
    """Neural network implementation with training and prediction capabilities."""
    # ... (NeuralNetwork class code as you provided) ...
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize weights and biases using Xavier initialization."""
        for i in range(len(self.layer_sizes)-1):
            scale = np.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i+1]))
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * scale)
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))

    def sigmoid(self, x):
        """Sigmoid activation function with clipping to prevent overflow."""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function for backpropagation."""
        return x * (1 - x)

    def forward(self, X):
        """Forward pass through the network."""
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights)-1:  # Output layer: softmax
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                activations.append(exp_z / exp_z.sum(axis=1, keepdims=True))
            else:  # Hidden layers: sigmoid
                activations.append(self.sigmoid(z))
        return activations

    def backward(self, X, y, activations, lr):
        """Backward pass to update weights and biases."""
        m = X.shape[0]
        deltas = []

        # Output layer delta (softmax cross-entropy derivative)
        delta = (activations[-1] - y.reshape(y.shape[0], -1))
        deltas.append(delta)

        # Hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            delta = np.dot(deltas[-1], self.weights[i+1].T) * self.sigmoid_derivative(activations[i+1])
            deltas.append(delta)
        deltas.reverse()

        # Update parameters
        for i in range(len(self.weights)):
            grad_w = np.dot(activations[i].T, deltas[i]) / m
            grad_b = np.mean(deltas[i], axis=0, keepdims=True)
            self.weights[i] -= lr * grad_w
            self.biases[i] -= lr * grad_b

        # Calculate cross-entropy loss
        log_probs = np.log(activations[-1] + 1e-8)
        return -np.sum(y.reshape(y.shape[0], -1) * log_probs) / m

    def train_epoch(self, X, y, lr=0.01, batch_size=128):
        """Train the model for one epoch using mini-batch gradient descent."""
        indices = np.random.permutation(X.shape[0])
        total_loss = 0

        for i in range(0, X.shape[0], batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch, y_batch = X[batch_idx], y[batch_idx]

            activations = self.forward(X_batch)
            loss = self.backward(X_batch, y_batch, activations, lr)
            total_loss += loss * X_batch.shape[0]

        avg_loss = total_loss / X.shape[0]
        return avg_loss

    def predict(self, X):
        """Predict class labels for input data."""
        return np.argmax(self.forward(X)[-1], axis=1)

    def save_model(self, filepath):
        """Saves the model parameters to a JSON file."""
        model_data = {
            "layer_sizes": [int(size) for size in self.layer_sizes], # Convert layer_sizes to list of int
            "weights": [[[float(val) for val in row] for row in w.tolist()] for w in self.weights], # Explicit float conversion
            "biases": [[[float(val) for val in row] for row in b.tolist()] for b in self.biases]   # Explicit float conversion
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model parameters from a JSON file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        self.layer_sizes = model_data["layer_sizes"]
        self.weights = [np.array(w) for w in model_data["weights"]]
        self.biases = [np.array(b) for b in model_data["biases"]]
        print(f"Model loaded from {filepath}")

# Constants (No changes needed)
FINAL_SIZE = 50
DEFAULT_CANVAS_SIZE = 400
DEFAULT_BRUSH_RADIUS = 15

# CanvasManager, ImagePreviewFrame, FileManager (No changes needed in these classes)
class CanvasManager:
    """Manages the drawing canvas and its operations"""
    # ... (CanvasManager class code as you provided) ...
    def __init__(self, parent, canvas_size=DEFAULT_CANVAS_SIZE, brush_radius_var=None):
        self.parent = parent
        self.canvas_size = canvas_size
        self.brush_radius_var = brush_radius_var  # Store the IntVar directly
        self.stroke_history = []
        self.bbox_id = None
        self.bbox_coords = None

        # Create the canvas
        self.canvas = tk.Canvas(
            parent, width=canvas_size, height=canvas_size,
            bg="white", bd=2, highlightthickness=2, highlightbackground="black"
        )

        # Create PIL image and drawing context
        self.image = Image.new("L", (canvas_size, canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Event bindings
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-3>", self.delete_stroke)

    def paint(self, event):
        """Draw on canvas and update internal image"""
        # Get brush radius directly from the IntVar
        r = self.brush_radius_var.get() if self.brush_radius_var else DEFAULT_BRUSH_RADIUS
        x, y = event.x, event.y
        oval_id = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.stroke_history.append((x - r, y - r, x + r, y + r, r, oval_id))
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")
        self.update_bbox()

    def delete_stroke(self, event):
        """Delete the closest stroke to the mouse pointer"""
        x, y = event.x, event.y
        if not self.stroke_history:
            return

        closest_stroke = min(
            self.stroke_history,
            key=lambda stroke: ((stroke[0] + stroke[2]) / 2 - x)**2 + ((stroke[1] + stroke[3]) / 2 - y)**2
        )

        self.canvas.delete(closest_stroke[-1])
        self.stroke_history.remove(closest_stroke)
        self.redraw_image()
        self.update_bbox()

    def update_bbox(self):
        """Update the bounding box around the drawing"""
        if self.bbox_id:
            self.canvas.delete(self.bbox_id)

        if self.stroke_history:
            x_coords = [stroke[0] for stroke in self.stroke_history] + [stroke[2] for stroke in self.stroke_history]
            y_coords = [stroke[1] for stroke in self.stroke_history] + [stroke[3] for stroke in self.stroke_history]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(self.canvas_size, x_max)
            y_max = min(self.canvas_size, y_max)
            self.bbox_coords = (x_min, y_min, x_max, y_max)
            self.bbox_id = self.canvas.create_rectangle(
                x_min, y_min, x_max, y_max,
                outline="red", width=2
            )
        else:
            self.bbox_coords = None
            self.bbox_id = None

    def undo(self):
        """Undo the last stroke"""
        if self.stroke_history:
            x1, y1, x2, y2, _, oval_id = self.stroke_history.pop()
            self.canvas.delete(oval_id)
            self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
            self.draw = ImageDraw.Draw(self.image)
            for stroke in self.stroke_history:
                x1, y1, x2, y2, _, _ = stroke
                self.draw.ellipse([x1, y1, x2, y2], fill="black")
            self.update_bbox()

    def redraw_image(self):
        """Redraw the entire image from stroke history"""
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        for x1, y1, x2, y2, radius, _ in self.stroke_history:
            self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def reset(self):
        """Reset canvas to blank state"""
        self.canvas.delete("all")
        self.stroke_history = []
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.bbox_id = None
        self.bbox_coords = None

    def resize(self, new_size):
        """Resize the canvas"""
        self.canvas_size = new_size
        self.canvas.config(width=new_size, height=new_size)
        self.reset()

    def get_processed_image(self):
        """Return processed image ready for saving or prediction"""
        if self.bbox_coords:
            x_min, y_min, x_max, y_max = self.bbox_coords
            cropped_image = self.image.crop((x_min, y_min, x_max, y_max))
            image_to_process = cropped_image
        else:
            image_to_process = self.image

        # Resize and invert for MNIST-like format
        return ImageOps.invert(image_to_process.resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS))

class ImagePreviewFrame:
    """Manages the preview of saved images"""
    # ... (ImagePreviewFrame class code as you provided) ...
    def __init__(self, parent):
        self.parent = parent
        self.preview_container = ttk.LabelFrame(parent, text="Last Saved Image")
        self.preview_label = tk.Label(self.preview_container, bd=2, relief="groove")
        self.photo_image = None

        # Layout
        self.preview_label.pack(padx=10, pady=10)

        # Initialize with blank image
        self.display_image()

    def display_image(self, image=None):
        """Display an image in the preview panel"""
        if image is None:
            img = Image.new("L", (FINAL_SIZE, FINAL_SIZE), "white")
        else:
            img = image

        # Ensure image is correctly sized
        img_resized = img.resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(img_resized)
        self.preview_label.config(image=self.photo_image)

    def pack(self, **kwargs):
        """Pack the preview container with given options"""
        self.preview_container.pack(**kwargs)

class PredictionManager:
    """Handles digit prediction functionality"""
    # ... (PredictionManager class code as you provided) ...
    def __init__(self, parent, prediction_label, controller, dataset): # ADD controller and dataset
        self.parent = parent
        self.prediction_label = prediction_label
        self.controller = controller # ADDED
        self.dataset = dataset # ADDED

    def predict(self, image):
        """Predict digit from image using the trained model"""
        if self.controller.model is None:
            messagebox.showerror("Error", "Model not trained or loaded yet. Please go to 'Neural Network Trainer' tab and train or load a model.")
            self.prediction_label.config(text="Prediction: Model not ready") # Update label even if error
            return None

        try:
            # 1. Preprocess the image to get features (same as training)
            feature_method = self.controller.dataset.feature_method # Corrected attribute name here
            feature_size = self.controller.dataset.current_feature_size # Get from controller dataset

            processed_image_np = np.array(image) / 255.0 # Normalize like in dataset loading

            features = self.dataset.image_to_features( # Use dataset instance to extract features
                processed_image_np,
                method=feature_method,
                feature_size=feature_size
            )
            features = features.reshape(1, -1) # Reshape to (1, num_features) for single prediction

            # 2. Get activations from forward pass
            activations = self.controller.model.forward(features) # Get all layer activations

            # 3. Use the model to predict (still need prediction for label)
            prediction = self.controller.model.predict(features)[0] # Get single prediction

            # 4. Update the prediction label
            self.prediction_label.config(text=f"Prediction: {prediction}")
            return features, activations # Return features and activations for plotting

        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {e}")
            self.prediction_label.config(text="Prediction: Error") # Update label in case of error
            return None, None # Return None, None in case of error


class FileManager:
    """Manages file operations and naming"""
    # ... (FileManager class code as you provided) ...
    def __init__(self, filename_var, auto_count, start_number):
        self.filename_var = filename_var
        self.auto_count = auto_count
        self.start_number = start_number
        self.last_saved_path = None

        # Create dataset directory if it doesn't exist
        os.makedirs("dataset", exist_ok=True)

    def save_image(self, image):
        """Save the processed image to file"""
        filename = self.filename_var.get()

        # Create filename based on settings
        if self.auto_count.get():
            file_id = f"{filename}_{self.start_number.get()}"
            # Increment counter after save
            self.start_number.set(self.start_number.get() + 1)
        else:
            file_id = f"{filename}_{int(time.time())}"

        # Complete path
        file_path = os.path.join("dataset", f"{file_id}.png")

        # Save the image
        image.save(file_path)
        self.last_saved_path = file_path

        print(f"Image saved as {file_path}")
        return file_path

class ControlPanel:
    """Manages all control widgets and settings"""
    # ... (ControlPanel class code as you provided) ...
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.plotter = None  # Initialize plotter here

        # Create frames for organized controls
        self.create_brush_settings()
        self.create_canvas_settings()
        self.create_file_management()
        self.create_auto_counter()
        self.create_action_buttons()

        # Create preview frame
        self.preview = ImagePreviewFrame(parent)
        self.preview.pack(fill="x", pady=5)

    def create_brush_settings(self):
        """Create brush settings controls"""
        brush_frame = ttk.LabelFrame(self.parent, text="Brush Settings")
        brush_frame.pack(fill="x", pady=5)

        tk.Scale(
            brush_frame, from_=1, to=20, orient=tk.HORIZONTAL,
            variable=self.app.brush_radius, label="Brush Radius:"
        ).pack(fill="x")

    def create_canvas_settings(self):
        """Create canvas settings controls"""
        canvas_frame = ttk.LabelFrame(self.parent, text="Canvas Settings")
        canvas_frame.pack(fill="x", pady=5)

        tk.Scale(
            canvas_frame, from_=200, to=600, orient=tk.HORIZONTAL,
            variable=self.app.canvas_size, label="Canvas Size:",
            command=self.app.resize_canvas
        ).pack(fill="x")

    def create_file_management(self):
        """Create file management controls"""
        file_frame = ttk.LabelFrame(self.parent, text="File Management")
        file_frame.pack(fill="x", pady=5)

        # Radio buttons for digits 0-9
        for i in range(10):
            digit_str = str(i)
            tk.Radiobutton(
                file_frame,
                text=digit_str,
                variable=self.app.filename_var,
                value=digit_str,
                indicatoron=0,
                width=3
            ).pack(side="left", padx=2, pady=2)

        # Set default to "unknown"
        self.app.filename_var.set("unknown")

    def create_auto_counter(self):
        """Create auto counter controls"""
        auto_frame = ttk.LabelFrame(self.parent, text="Auto Counting")
        auto_frame.pack(fill="x", pady=5)

        tk.Checkbutton(
            auto_frame, text="Enable Auto-count", variable=self.app.auto_count,
            command=self.toggle_auto_count
        ).pack(side="left", padx=5)

        self.spinbox = tk.Spinbox(
            auto_frame, from_=1, to=1000, textvariable=self.app.start_number,
            state="disabled", width=8
        )
        self.spinbox.pack(side="right", padx=5)

    def toggle_auto_count(self):
        """Toggle auto count spinbox state"""
        self.spinbox.config(state='normal' if self.app.auto_count.get() else 'disabled')

    def create_action_buttons(self):
        """Create action buttons"""
        action_frame = ttk.LabelFrame(self.parent, text="Actions")
        action_frame.pack(fill="x", pady=5)

        # Configure grid layout
        for i in range(5): # Increased to 5 columns to include "Model Preview"
            action_frame.grid_columnconfigure(i, weight=1)

        # Create buttons
        tk.Button(action_frame, text="Undo (d)", command=self.app.undo).grid(
            row=0, column=0, sticky="ew", padx=2, pady=2
        )
        tk.Button(action_frame, text="Save (s)", command=self.app.save_image).grid(
            row=0, column=1, sticky="ew", padx=2, pady=2
        )
        tk.Button(action_frame, text="Reset (r)", command=self.app.reset_canvas).grid(
            row=0, column=2, sticky="ew", padx=2, pady=2
        )
        tk.Button(action_frame, text="Predict", command=self.app.predict_digit).grid(
            row=0, column=3, sticky="ew", padx=2, pady=2
        )
        tk.Button(action_frame, text="Model Preview", command=self.app.open_model_preview_popup).grid( # Added button
            row=0, column=4, sticky="ew", padx=2, pady=2
        )


        # Create prediction label
        self.prediction_label = tk.Label(action_frame, text="Prediction: ")
        self.prediction_label.grid(row=1, column=0, columnspan=5, sticky="ew", padx=2, pady=5) # Updated columnspan

        # Set prediction label
        self.app.prediction_label = self.prediction_label

class DrawingAppTab:
    """Drawing App as a Tab"""

    def __init__(self, parent, nn_controller, nn_dataset): # ADDED nn_controller and nn_dataset
        self.parent = parent
        self.nn_controller = nn_controller # ADDED
        self.nn_dataset = nn_dataset # ADDED
        self.plotter = None # Initialize plotter here, will be used in popup

        # Initialize variables
        self.canvas_size = tk.IntVar(value=DEFAULT_CANVAS_SIZE)
        self.brush_radius = tk.IntVar(value=DEFAULT_BRUSH_RADIUS)
        self.auto_count = tk.BooleanVar()
        self.start_number = tk.IntVar(value=1)
        self.filename_var = tk.StringVar()
        self.prediction_label = None
        self.prediction_manager = None  # Will be initialized after control panel creates label
        self.plotter = NeuralNetworkPlotter(None, None) # Initialize plotter here without Figure and Axes


        # Configure layout
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        # Create control panel (left)
        control_frame = ttk.Frame(parent, padding=10)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.control_panel = ControlPanel(control_frame, self)
        self.plotter = self.control_panel.plotter # Initialize plotter in DrawingAppTab

        # Create canvas container (right)
        canvas_container = ttk.Frame(parent, padding=10)
        canvas_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Create canvas manager - pass the brush_radius variable directly
        self.canvas_manager = CanvasManager(
            canvas_container,
            self.canvas_size.get(),
            brush_radius_var=self.brush_radius  # Pass the IntVar directly
        )
        self.canvas_manager.canvas.pack(pady=10)

        # Create file manager
        self.file_manager = FileManager(self.filename_var, self.auto_count, self.start_number)

        # Create prediction manager (after prediction_label is set by control panel)
        self.prediction_manager = None  # Will be initialized after control panel creates label


    def get_brush_radius(self):
        """Get current brush radius"""
        return self.brush_radius.get()

    def resize_canvas(self, *args):
        """Resize canvas based on slider"""
        self.canvas_manager.resize(self.canvas_size.get())

    def undo(self):
        """Undo last stroke"""
        self.canvas_manager.undo()

    def reset_canvas(self):
        """Reset canvas to blank state"""
        self.canvas_manager.reset()

    def save_image(self):
        """Process and save current image"""
        # Get processed image
        processed_image = self.canvas_manager.get_processed_image()

        # Save image
        self.file_manager.save_image(processed_image)

        # Update preview
        self.control_panel.preview.display_image(processed_image)

        # Reset canvas
        self.canvas_manager.reset()


    def predict_digit(self):
        """Predict digit from current drawing"""
        # Initialize prediction manager if needed
        if self.prediction_manager is None and self.prediction_label is not None:
            self.prediction_manager = PredictionManager(self, self.prediction_label, self.nn_controller, self.nn_dataset) # ADDED controller and dataset

        # Get processed image
        processed_image = self.canvas_manager.get_processed_image()

        # Update preview
        self.control_panel.preview.display_image(processed_image)

        # Predict digit and get features and activations
        prediction_result = self.prediction_manager.predict(processed_image) # Get prediction result
        if prediction_result: # Check if prediction was successful
            features, activations = prediction_result # Unpack if successful
            # Store features and activations for preview later
            self.last_features = features
            self.last_activations = activations

        # Reset canvas
        self.canvas_manager.reset()

        # Increment counter if auto_count is enabled
        if self.auto_count.get():
            self.start_number.set(self.start_number.get() + 1)

    def open_model_preview_popup(self):
        """Opens a popup window to display the neural network model plot with input and output values."""
        if self.nn_controller.model is None:
            messagebox.showinfo("Info", "Model not created yet. Please create a model in 'Neural Network Trainer' tab first.")
            return

        layer_sizes = self.nn_controller.model.layer_sizes
        if not layer_sizes:
            messagebox.showerror("Error", "Could not retrieve model layer sizes.")
            return

        if not hasattr(self, 'last_features') or not hasattr(self, 'last_activations'):
            messagebox.showinfo("Info", "Please make a prediction first to preview model output.")
            return

        popup = tk.Toplevel(self.parent)
        popup.title("Neural Network Model Preview with Values")

        plot_fig = Figure(figsize=(7, 5), dpi=100) # Adjust figsize as needed for popup
        plot_ax = plot_fig.add_subplot(111)
        plotter = NeuralNetworkPlotter(plot_fig, plot_ax) # Create plotter for popup

        input_values = self.last_features.flatten() if self.last_features is not None else None # Use stored features
        output_values = self.last_activations[-1].flatten() if self.last_activations is not None else None # Use stored output activations

        plotter.plot(layer_sizes, input_values=input_values, output_values=output_values) # Plot with values
        canvas = FigureCanvasTkAgg(plot_fig, master=popup)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()




if __name__ == "__main__":
    app = MergedApp()
    app.mainloop()
