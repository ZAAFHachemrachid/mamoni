"""
Controls Module - Manages control panel widgets and settings
"""
import tkinter as tk
from tkinter import ttk

from .preview import ImagePreviewFrame

class ControlPanel:
    """Manages all control widgets and settings"""
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.plotter = None

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
        for i in range(5):  # Increased to 5 columns to include "Model Preview"
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
        tk.Button(action_frame, text="Model Preview", command=self.app.open_model_preview_popup).grid(
            row=0, column=4, sticky="ew", padx=2, pady=2
        )

        # Create prediction label
        self.prediction_label = tk.Label(action_frame, text="Prediction: ")
        self.prediction_label.grid(row=1, column=0, columnspan=5, sticky="ew", padx=2, pady=5)

        # Set prediction label
        self.app.prediction_label = self.prediction_label