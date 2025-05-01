"""
Controls Module - Manages control panel widgets and settings
"""
import customtkinter as ctk

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
        brush_frame = ctk.CTkFrame(self.parent)
        brush_frame.pack(fill="x", pady=5, padx=5)
        
        # Add title label
        ctk.CTkLabel(brush_frame, text="Brush Settings", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 0))
        
        # Brush radius slider
        ctk.CTkSlider(
            brush_frame,
            from_=1,
            to=20,
            variable=self.app.brush_radius,
            number_of_steps=19
        ).pack(fill="x", padx=10, pady=10)
        
        # Label for the slider
        ctk.CTkLabel(brush_frame, text=f"Brush Radius: {self.app.brush_radius.get()}").pack(pady=(0, 5))

    def create_canvas_settings(self):
        """Create canvas settings controls"""
        canvas_frame = ctk.CTkFrame(self.parent)
        canvas_frame.pack(fill="x", pady=5, padx=5)
        
        # Add title label
        ctk.CTkLabel(canvas_frame, text="Canvas Settings", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 0))
        
        # Canvas size slider
        ctk.CTkSlider(
            canvas_frame,
            from_=200,
            to=600,
            variable=self.app.canvas_size,
            command=self.app.resize_canvas,
            number_of_steps=40
        ).pack(fill="x", padx=10, pady=10)
        
        # Label for the slider
        ctk.CTkLabel(canvas_frame, text=f"Canvas Size: {self.app.canvas_size.get()}").pack(pady=(0, 5))

    def create_file_management(self):
        """Create file management controls"""
        file_frame = ctk.CTkFrame(self.parent)
        file_frame.pack(fill="x", pady=5, padx=5)
        
        # Add title label
        ctk.CTkLabel(file_frame, text="File Management", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 0))
        
        # Create frame for radio buttons
        radio_frame = ctk.CTkFrame(file_frame)
        radio_frame.pack(fill="x", padx=5, pady=5)
        
        # Radio buttons for digits 0-9
        for i in range(10):
            digit_str = str(i)
            ctk.CTkButton(
                radio_frame,
                text=digit_str,
                width=30,
                command=lambda x=digit_str: self.app.filename_var.set(x)
            ).pack(side="left", padx=2, pady=2)

        # Set default to "unknown"
        self.app.filename_var.set("unknown")

    def create_auto_counter(self):
        """Create auto counter controls"""
        auto_frame = ctk.CTkFrame(self.parent)
        auto_frame.pack(fill="x", pady=5, padx=5)
        
        # Add title label
        ctk.CTkLabel(auto_frame, text="Auto Counting", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 0))
        
        # Create inner frame for controls
        control_frame = ctk.CTkFrame(auto_frame)
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # Checkbox
        ctk.CTkCheckBox(
            control_frame,
            text="Enable Auto-count",
            variable=self.app.auto_count,
            command=self.toggle_auto_count
        ).pack(side="left", padx=5)
        
        # Entry (replacement for Spinbox)
        self.spinbox = ctk.CTkEntry(
            control_frame,
            textvariable=self.app.start_number,
            width=80,
            state="disabled"
        )
        self.spinbox.pack(side="right", padx=5)

    def toggle_auto_count(self):
        """Toggle auto count entry state"""
        self.spinbox.configure(state='normal' if self.app.auto_count.get() else 'disabled')

    def create_action_buttons(self):
        """Create action buttons"""
        action_frame = ctk.CTkFrame(self.parent)
        action_frame.pack(fill="x", pady=5, padx=5)
        
        # Add title label
        ctk.CTkLabel(action_frame, text="Actions", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 0))
        
        # Create button frame
        button_frame = ctk.CTkFrame(action_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        # Configure grid layout
        for i in range(5):
            button_frame.grid_columnconfigure(i, weight=1)

        # Create buttons
        ctk.CTkButton(button_frame, text="Undo (d)", command=self.app.undo).grid(
            row=0, column=0, sticky="ew", padx=2, pady=2
        )
        ctk.CTkButton(button_frame, text="Save (s)", command=self.app.save_image).grid(
            row=0, column=1, sticky="ew", padx=2, pady=2
        )
        ctk.CTkButton(button_frame, text="Reset (r)", command=self.app.reset_canvas).grid(
            row=0, column=2, sticky="ew", padx=2, pady=2
        )
        ctk.CTkButton(button_frame, text="Predict", command=self.app.predict_digit).grid(
            row=0, column=3, sticky="ew", padx=2, pady=2
        )
        ctk.CTkButton(button_frame, text="Model Preview", command=self.app.open_model_preview_popup).grid(
            row=0, column=4, sticky="ew", padx=2, pady=2
        )

        # Create prediction label
        self.prediction_label = ctk.CTkLabel(action_frame, text="Prediction: ")
        self.prediction_label.pack(pady=5)

        # Set prediction label
        self.app.prediction_label = self.prediction_label