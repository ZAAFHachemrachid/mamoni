"""
Training Tab Module - Implements the neural network training interface
"""
import customtkinter as ctk
from tkinter import filedialog, messagebox
import time
import numpy as np

from core.neural_network import NeuralNetwork
from core.controller import NeuralNetController
from visualization.components import TrainingMetrics
from utils.progress import ProgressManager

class TrainingTab(ctk.CTkFrame):
    """Neural Network Training Application Tab"""

    def __init__(self, parent, data_prep_tab):
        super().__init__(parent)
        self.root = parent
        self.data_prep_tab = data_prep_tab
        self.dataset = data_prep_tab.get_dataset()
        self.controller = NeuralNetController(self.dataset)
        
        # Configuration variables
        self.hidden_layers_var = ctk.StringVar(value='64')
        self.epochs_var = ctk.StringVar(value="20")
        self.learning_rate_var = ctk.StringVar(value="0.1")
        self.batch_size_var = ctk.StringVar(value="128")

        self._init_ui()

    def _init_ui(self):
        """Initialize all UI components."""
        # Train Frame
        self._create_train_frame()
        self.train_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Progress Frame
        self._create_progress_frame()
        self.progress_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Configure grid weights for resizing
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def _create_train_frame(self):
        """Creates the Training Frame."""
        self.train_frame = ctk.CTkFrame(self)

        # Metrics Plot Frame
        self.metrics_plot = TrainingMetrics(self.train_frame)
        self.metrics_plot.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Hidden Layer Frame
        self._create_hidden_layer_frame(row_num=1)
        
        # Training Parameters Frame
        self._create_training_params_frame(row_num=2)

        # Training Buttons Frame
        train_buttons_frame = ctk.CTkFrame(self.train_frame)
        train_buttons_frame.grid(row=3, column=0, sticky="ew", pady=5)
        
        buttons = [
            ("Create Model", self._create_model),
            ("Train Model", self._train_model),
            ("Test Model", self._test_model)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ctk.CTkButton(train_buttons_frame, text=text, command=command)
            btn.grid(row=0, column=i, sticky="ew", padx=2, pady=5)

        # Model I/O Buttons Frame
        model_io_frame = ctk.CTkFrame(self.train_frame)
        model_io_frame.grid(row=4, column=0, sticky="ew", pady=5)
        
        io_buttons = [
            ("Save Model", self._save_model),
            ("Load Model", self._load_model)
        ]
        
        for i, (text, command) in enumerate(io_buttons):
            btn = ctk.CTkButton(model_io_frame, text=text, command=command)
            btn.grid(row=0, column=i, sticky="ew", padx=2)

        self.train_frame.grid_columnconfigure(0, weight=1)

    def _create_progress_frame(self):
        """Creates the Progress Bar Frame."""
        self.progress_frame = ctk.CTkFrame(self)
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.set(0)  # Initialize progress to 0
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="")
        self.progress_label.pack(pady=5)
        self.progress_manager = ProgressManager(self.progress_bar, self.progress_label, self.root)

    def _create_hidden_layer_frame(self, row_num):
        """Creates the hidden layer size input frame."""
        hidden_layer_frame = ctk.CTkFrame(self.train_frame)
        hidden_layer_frame.grid(row=row_num, column=0, sticky="ew", pady=5)
        ctk.CTkLabel(hidden_layer_frame, text="Hidden Layer Sizes:").grid(row=0, column=0, sticky="w", padx=5)
        hidden_layers_entry = ctk.CTkEntry(hidden_layer_frame, textvariable=self.hidden_layers_var, width=250)
        hidden_layers_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        hidden_layer_frame.grid_columnconfigure(1, weight=1)

    def _create_training_params_frame(self, row_num):
        """Create the training parameters section."""
        training_params_frame = ctk.CTkFrame(self.train_frame)
        training_params_frame.grid(row=row_num, column=0, sticky="ew", pady=5)
        ctk.CTkLabel(training_params_frame, text="Training Parameters:").grid(row=0, column=0, columnspan=6, sticky="w", padx=5, pady=5)

        params = [
            ("Epochs:", self.epochs_var),
            ("Learning Rate:", self.learning_rate_var),
            ("Batch Size:", self.batch_size_var)
        ]

        for i, (label_text, var) in enumerate(params):
            ctk.CTkLabel(training_params_frame, text=label_text).grid(row=1, column=i*2, sticky="w", padx=5)
            entry = ctk.CTkEntry(training_params_frame, textvariable=var, width=80)
            entry.grid(row=1, column=i*2+1, sticky="w", padx=5)

    def _create_model(self):
        """Create model after data prepared."""
        if self.dataset.features is None:
            messagebox.showerror("Error", "Dataset features not prepared. Please prepare data first.")
            return

        try:
            self.progress_manager.start(mode='indeterminate', text="Creating Model...")

            hidden_layers_str = self.hidden_layers_var.get()
            hidden_layer_sizes = [int(size) for size in hidden_layers_str.split(',') if size.strip()]
            feature_size = self.data_prep_tab.get_feature_size()
            input_size = np.prod(feature_size)
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

    def _load_model(self):
        """Load a model from a file."""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.progress_manager.start(mode='indeterminate', text="Loading Model...")
                feature_size = self.data_prep_tab.get_feature_size()
                input_size = np.prod(feature_size)
                self.controller.load_model(filepath, input_size=input_size)
                self.progress_manager.stop(text="Model Loaded")
                messagebox.showinfo("Info", "Model loaded successfully!")
            except Exception as e:
                self.progress_manager.stop(text="Load Failed")
                messagebox.showerror("Error", f"Error loading model: {e}")