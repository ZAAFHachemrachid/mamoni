"""
Drawing Tab Module - Implements the drawing interface tab
"""
import tkinter as tk
from tkinter import messagebox
import os

from ..canvas import CanvasManager
from ..controls import ControlPanel
from core.controller import NeuralNetController
from utils.file_manager import FileManager
from utils.prediction import PredictionManager

class DrawingAppTab(tk.Frame):
    """Drawing App as a Tab"""

    def __init__(self, parent, nn_controller, nn_dataset):
        super().__init__(parent)
        self.parent = parent
        self.controller = nn_controller
        self.dataset = nn_dataset

        # Variables for canvas and UI control
        self.brush_radius = tk.IntVar(value=15)
        self.canvas_size = tk.IntVar(value=400)
        self.filename_var = tk.StringVar()
        self.auto_count = tk.BooleanVar()
        self.start_number = tk.IntVar(value=1)
        self.prediction_label = None

        # Initialize FileManager for saving images
        self.file_manager = FileManager(self.filename_var, self.auto_count, self.start_number)

        # Create UI layout
        self._create_layout()

    def _create_layout(self):
        """Create the main layout"""
        # Left side - Canvas
        canvas_frame = tk.Frame(self)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create canvas manager
        self.canvas_manager = CanvasManager(canvas_frame, self.canvas_size.get(), self.brush_radius)
        self.canvas_manager.canvas.pack(expand=True)

        # Right side - Controls
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Create control panel
        self.control_panel = ControlPanel(control_frame, self)

        # Create prediction manager
        self.prediction_manager = PredictionManager(self, self.prediction_label, self.controller, self.dataset)

        # Key bindings
        self.bind_keys()

    def bind_keys(self):
        """Set up key bindings"""
        self.bind_all('d', lambda e: self.undo())
        self.bind_all('s', lambda e: self.save_image())
        self.bind_all('r', lambda e: self.reset_canvas())

    def undo(self):
        """Undo last drawing action"""
        self.canvas_manager.undo()

    def reset_canvas(self):
        """Reset the canvas"""
        self.canvas_manager.reset()

    def resize_canvas(self, _):
        """Resize the canvas"""
        self.canvas_manager.resize(self.canvas_size.get())

    def save_image(self):
        """Save the current drawing"""
        if not any(str(i) == self.filename_var.get() for i in range(10)) and self.filename_var.get() != "unknown":
            messagebox.showerror("Error", "Please select a digit (0-9) or 'unknown'")
            return

        processed_image = self.canvas_manager.get_processed_image()
        saved_path = self.file_manager.save_image(processed_image)
        
        if saved_path:
            self.control_panel.preview.display_image(processed_image)
            # Predict the digit if it was saved
            self.predict_digit()
            # Reset canvas after successful save
            self.reset_canvas()

    def predict_digit(self):
        """Predict the drawn digit"""
        processed_image = self.canvas_manager.get_processed_image()
        self.prediction_manager.predict(processed_image)

    def open_model_preview_popup(self):
        """Open the model preview window"""
        if self.controller and self.controller.model:
            if hasattr(self, 'plotter') and self.plotter:
                self.plotter.destroy()
            
            self.plotter = tk.Toplevel(self)
            self.plotter.title("Neural Network Preview")
            from visualization.plotnn import NetworkPlotter
            NetworkPlotter(self.plotter, self.controller.model)
        else:
            messagebox.showerror("Error", "No model loaded")