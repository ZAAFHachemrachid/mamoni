"""
Training Tab Module - Implements the neural network training interface
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time
import numpy as np

from data.data_layer import ImageDataset
from core.neural_network import NeuralNetwork
from core.controller import NeuralNetController
from visualization.components import AnimatedHeatmap, TrainingMetrics
from visualization.plotnn import NeuralNetworkPlotter
from ..tooltip import ToolTip
from utils.progress import ProgressManager

class NeuralNetworkAppTab(ttk.Frame):
    """Neural Network Training Application as a Tab"""

    def __init__(self, parent):
        super().__init__(parent)
        self.root = parent
        self.dataset = ImageDataset()
        self.controller = NeuralNetController(self.dataset)
        
        # Configuration variables
        self.feature_method_var = tk.StringVar(value='average')
        self.feature_size_var = tk.StringVar(value='5x5')
        self.hidden_layers_var = tk.StringVar(value='64')
        self.train_ratio_var = tk.DoubleVar(value=0.7)
        self.val_ratio_var = tk.DoubleVar(value=0.15)
        self.test_ratio_var = tk.DoubleVar(value=0.15)
        self.dataset_path_var = tk.StringVar(value=r"D:\python\tp_nn_final_gui\MyData")
        self.images_per_class_var = tk.StringVar(value="10")
        self.epochs_var = tk.StringVar(value="20")
        self.learning_rate_var = tk.StringVar(value="0.1")
        self.batch_size_var = tk.StringVar(value="128")

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
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def _create_data_frame(self, parent):
        """Creates the Data Processing Frame."""
        self.data_frame = ttk.Frame(parent, padding=10)

        # Heatmap Frame (inside Data Frame)
        heatmap_params = {'data_shape': (50, 50), 'cmap': 'gray', 'vmin': 0, 'vmax': 255, 'figsize': (3, 3)}
        self.heatmap = AnimatedHeatmap(self.data_frame, params=heatmap_params)
        self.heatmap.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Dataset Path Frame
        dataset_path_frame = ttk.Frame(self.data_frame)
        dataset_path_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(dataset_path_frame, text="Dataset Path:").grid(row=0, column=0, sticky=tk.W, padx=5)
        dataset_path_entry = ttk.Entry(dataset_path_frame, textvariable=self.dataset_path_var, width=30)
        dataset_path_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        dataset_path_button = ttk.Button(dataset_path_frame, text="Browse", command=self._browse_dataset_path)
        dataset_path_button.grid(row=0, column=2, sticky=tk.W, padx=5)

        # Tooltips
        ToolTip(dataset_path_button, "Browse your dataset directory")
        ToolTip(dataset_path_entry, "Path to the dataset directory")

        # Images Per Class Frame
        images_per_class_frame = ttk.Frame(self.data_frame)
        images_per_class_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(images_per_class_frame, text="Images/Class:").grid(row=0, column=0, sticky=tk.W, padx=5)
        images_per_class_entry = ttk.Entry(images_per_class_frame, textvariable=self.images_per_class_var, width=5)
        images_per_class_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        ToolTip(images_per_class_entry, "Max images to load per class")

        # Feature Method and Size Frames
        self._create_feature_method_frame(self.data_frame, row_num=3)
        self._create_feature_size_frame(self.data_frame, row_num=4)

        # Data Buttons Frame
        data_buttons_frame = ttk.Frame(self.data_frame)
        data_buttons_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=5)
        
        self.load_dataset_button = ttk.Button(data_buttons_frame, text="Load Dataset", command=self._load_dataset)
        self.load_dataset_button.grid(row=0, column=0, sticky="ew", padx=2, pady=5)
        
        self.prepare_data_button = ttk.Button(data_buttons_frame, text="Prepare Data", command=self._prepare_data)
        self.prepare_data_button.grid(row=0, column=1, sticky="ew", padx=2, pady=5)
        
        # Tooltips
        ToolTip(self.load_dataset_button, "Load dataset from directory")
        ToolTip(self.prepare_data_button, "Prepare features from loaded dataset")

        self.data_frame.grid_columnconfigure(0, weight=1)

    def _create_train_frame(self, parent):
        """Creates the Training Frame."""
        self.train_frame = ttk.Frame(parent, padding=10)

        # Metrics Plot Frame (inside Train Frame)
        self.metrics_plot = TrainingMetrics(self.train_frame)
        self.metrics_plot.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Training configuration frames
        self._create_hidden_layer_frame(self.train_frame, row_num=1)
        self._create_dataset_split_frame(self.train_frame, row_num=2)
        self._create_training_params_frame(self.train_frame, row_num=3)

        # Training Buttons Frame
        train_buttons_frame = ttk.Frame(self.train_frame)
        train_buttons_frame.grid(row=4, column=0, sticky="ew", pady=5)
        
        buttons = [
            ("Create Model", self._create_model),
            ("Train Model", self._train_model),
            ("Test Model", self._test_model)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ttk.Button(train_buttons_frame, text=text, command=command)
            btn.grid(row=0, column=i, sticky="ew", padx=2, pady=5)
            ToolTip(btn, f"{text}")

        # Model I/O Buttons Frame
        model_io_frame = ttk.Frame(self.train_frame)
        model_io_frame.grid(row=5, column=0, sticky="ew", pady=5)
        
        io_buttons = [
            ("Save Model", self._save_model),
            ("Load Model", self._load_model),
            ("Export Features", self._export_features)
        ]
        
        for i, (text, command) in enumerate(io_buttons):
            btn = ttk.Button(model_io_frame, text=text, command=command)
            btn.grid(row=0, column=i, sticky="ew", padx=2)
            ToolTip(btn, f"{text}")

        self.train_frame.grid_columnconfigure(0, weight=1)

    def _create_progress_frame(self):
        """Creates the Progress Bar Frame."""
        self.progress_frame = ttk.Frame(self, padding=10)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.progress_label = ttk.Label(self.progress_frame, text="")
        self.progress_label.pack(pady=5)
        self.progress_manager = ProgressManager(self.progress_bar, self.progress_label, self.root)

    def _create_feature_method_frame(self, parent_frame, row_num):
        """Creates the feature method selection frame."""
        feature_method_frame = ttk.LabelFrame(parent_frame, text="Feature Method", padding=5)
        feature_method_frame.grid(row=row_num, column=0, columnspan=2, sticky="ew", pady=5)

        methods = ['average', 'sum', 'max']
        for i, method in enumerate(methods):
            rb = ttk.Radiobutton(feature_method_frame, text=method.capitalize(),
                               variable=self.feature_method_var, value=method)
            rb.grid(row=0, column=i, padx=10, pady=5, sticky='w')
            ToolTip(rb, f"Select {method.capitalize()} pooling")

    def _create_feature_size_frame(self, parent_frame, row_num):
        """Creates the feature size selection frame."""
        feature_size_frame = ttk.LabelFrame(parent_frame, text="Feature Size", padding=5)
        feature_size_frame.grid(row=row_num, column=0, columnspan=2, sticky="ew", pady=5)

        sizes = ['5x5', '10x10', '25x25', '50x50 (No Prep)']
        for i, size in enumerate(sizes):
            rb = ttk.Radiobutton(feature_size_frame, text=size,
                               variable=self.feature_size_var, value=size)
            rb.grid(row=0, column=i, padx=10, pady=5, sticky='w')
            ToolTip(rb, f"Select feature size: {size}")

    def _create_hidden_layer_frame(self, parent_frame, row_num):
        """Creates the hidden layer size input frame."""
        hidden_layer_frame = ttk.LabelFrame(parent_frame, text="Hidden Layers", padding=5)
        hidden_layer_frame.grid(row=row_num, column=0, sticky="ew", pady=5)

        ttk.Label(hidden_layer_frame, text="Sizes (comma-separated):").grid(row=0, column=0, sticky=tk.W, padx=5)
        hidden_layers_entry = ttk.Entry(hidden_layer_frame, textvariable=self.hidden_layers_var, width=30)
        hidden_layers_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        hidden_layer_frame.grid_columnconfigure(1, weight=1)
        ToolTip(hidden_layers_entry, "Hidden layer sizes, e.g., '64,32'")

    def _create_dataset_split_frame(self, parent_frame, row_num):
        """Creates the dataset split ratio frame."""
        split_frame = ttk.LabelFrame(parent_frame, text="Dataset Split Ratios", padding=5)
        split_frame.grid(row=row_num, column=0, sticky="ew", pady=5)

        entries = [
            ("Train Ratio:", self.train_ratio_var, "Train dataset ratio (0.0-1.0)"),
            ("Validation Ratio:", self.val_ratio_var, "Validation dataset ratio (0.0-1.0)"),
            ("Test Ratio:", self.test_ratio_var, "Test dataset ratio (0.0-1.0)")
        ]

        for i, (label_text, var, tooltip_text) in enumerate(entries):
            ttk.Label(split_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=5)
            entry = ttk.Entry(split_frame, textvariable=var, width=5)
            entry.grid(row=i, column=1, sticky="ew", padx=5)
            entry.bind("<FocusOut>", self._validate_and_normalize_ratios)
            ToolTip(entry, tooltip_text)

        split_frame.grid_columnconfigure(1, weight=1)
        self._validate_and_normalize_ratios()

    def _create_training_params_frame(self, parent_frame, row_num):
        """Create the training parameters section."""
        training_params_frame = ttk.LabelFrame(parent_frame, text="Training Parameters", padding=5)
        training_params_frame.grid(row=row_num, column=0, sticky="ew", pady=5)

        params = [
            ("Epochs:", self.epochs_var, "Number of epochs"),
            ("Learning Rate:", self.learning_rate_var, "Learning rate"),
            ("Batch Size:", self.batch_size_var, "Batch size")
        ]

        for i, (label_text, var, tooltip_text) in enumerate(params):
            ttk.Label(training_params_frame, text=label_text).grid(row=0, column=i*2, sticky=tk.W, padx=5)
            entry = ttk.Entry(training_params_frame, textvariable=var, width=5)
            entry.grid(row=0, column=i*2+1, sticky=tk.W, padx=5)
            ToolTip(entry, tooltip_text)

    def _validate_and_normalize_ratios(self, event=None):
        """Validates ratio entries and normalizes them to sum to 1.0."""
        try:
            train_ratio = float(self.train_ratio_var.get())
            val_ratio = float(self.val_ratio_var.get())
            test_ratio = float(self.test_ratio_var.get())

            if any(r < 0 or r > 1 for r in [train_ratio, val_ratio, test_ratio]):
                messagebox.showerror("Error", "Ratios must be between 0.0 and 1.0.")
                return

            total_ratio = train_ratio + val_ratio + test_ratio

            if not np.isclose(total_ratio, 1.0):
                if total_ratio > 0:
                    train_ratio /= total_ratio
                    val_ratio /= total_ratio
                    test_ratio /= total_ratio
                else:
                    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

                self.train_ratio_var.set(f"{train_ratio:.2f}")
                self.val_ratio_var.set(f"{val_ratio:.2f}")
                self.test_ratio_var.set(f"{test_ratio:.2f}")

        except ValueError:
            messagebox.showerror("Error", "Invalid ratio value. Please enter numbers between 0.0 and 1.0.")

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
            feature_method = self.feature_method_var.get()
            feature_size_str = self.feature_size_var.get()
            
            if feature_size_str == '5x5':
                feature_size = (5, 5)
            elif feature_size_str == '10x10':
                feature_size = (10, 10)
            elif feature_size_str == '25x25':
                feature_size = (25, 25)
            elif feature_size_str == '50x50 (No Prep)':
                feature_size = (50, 50)
            else:
                feature_size = (5, 5)

            self.dataset.prepare_features(
                progress_callback=self.progress_manager.update,
                heatmap_callback=self.heatmap.update_heatmap,
                feature_method=feature_method,
                feature_size=feature_size
            )

            train_ratio = self.train_ratio_var.get()
            val_ratio = self.val_ratio_var.get()
            test_ratio = self.test_ratio_var.get()
            self.dataset.split_dataset(train_ratio, val_ratio, test_ratio)

            self.progress_manager.stop(text="Data Prepared and Split")
            messagebox.showinfo("Info", f"Dataset features prepared successfully using {feature_method.capitalize()} "
                              f"method and feature size {feature_size_str}, and dataset split!")

        except ValueError as e:
            self.progress_manager.stop(text="Preparation Failed")
            messagebox.showerror("Error", str(e))
        except Exception as e:
            self.progress_manager.stop(text="Preparation Failed")
            messagebox.showerror("Error", f"Data preparation error: {e}")

    def _create_model(self):
        """Create model after data prepared to know input size."""
        if self.dataset.features is None:
            messagebox.showerror("Error", "Dataset features not prepared. Prepare data first.")
            return

        try:
            self.progress_manager.start(mode='indeterminate', text="Creating Model...")

            hidden_layers_str = self.hidden_layers_var.get()
            hidden_layer_sizes = [int(size) for size in hidden_layers_str.split(',') if size.strip()]
            feature_size = self.dataset.current_feature_size
            input_size = np.prod(feature_size) if feature_size != (50,50) else 50*50
            layer_sizes = [input_size] + hidden_layer_sizes + [self.dataset.num_classes]
            self.controller.create_model(layer_sizes=layer_sizes)

            self.progress_manager.stop(text="Model Created")
            messagebox.showinfo("Info", "Model Created successfully")

        except ValueError as e:
            self.progress_manager.stop(text="Create Model Failed")
            messagebox.showerror("Error", str(e))
        except Exception as e:
            self.progress_manager.stop(text="Create Model Failed")
            messagebox.showerror("Error", f"Model creating error: {e}")

    def _train_model(self):
        """Train the neural network model."""
        if self.controller.model is None:
            messagebox.showerror("Error", "Model not initialized. Create model first.")
            return
        if self.dataset.train_features is None:
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
                val_loss, val_accuracy = self.controller.validate()
                self.metrics_plot.update_plot(epoch + 1, loss, accuracy, val_loss, val_accuracy)
                self.progress_manager.update(
                    epoch + 1,
                    epochs,
                    text=f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                         f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                )
                self.root.update()
                time.sleep(0.01)

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
        if self.dataset.test_features is None:
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

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
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
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.progress_manager.start(mode='indeterminate', text="Loading Model...")
                feature_size = self.dataset.current_feature_size
                input_size = np.prod(feature_size) if feature_size != (50,50) else 50*50
                self.controller.load_model(filepath, input_size=input_size)
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

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
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