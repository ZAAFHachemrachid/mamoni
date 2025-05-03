"""
Data Preparation Tab Module

Handles all data loading and preparation functionality including:
- Dataset configuration
- Data loading and processing
- Feature extraction 
- Dataset splitting
"""
import os
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from visualization.components import AnimatedHeatmap
from data.data_layer import ImageDataset
from utils.progress import ProgressManager
from ..tooltip import ToolTip

class DataPreparationTab(ctk.CTkFrame):
    """Handles data loading and preparation functionality"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.root = parent
        self.dataset = ImageDataset()
        
        # Configuration variables
        self.feature_method_var = ctk.StringVar(value='average')
        self.feature_size_var = ctk.StringVar(value='5x5')
        self.dataset_path_var = ctk.StringVar(value="")
        self.images_per_class_var = ctk.StringVar(value="10")
        self.train_ratio_var = ctk.DoubleVar(value=0.7)
        self.val_ratio_var = ctk.DoubleVar(value=0.15)
        self.test_ratio_var = ctk.DoubleVar(value=0.15)
        
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI components"""
        # Left sidebar for controls
        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        
        # Right frame for visualization
        self.viz_frame = ctk.CTkFrame(self)
        self.viz_frame.grid(row=0, column=1, padx=5, pady=10, sticky="nsew")
        
        # Progress Frame at bottom
        self._create_progress_frame()
        self.progress_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        
        # Configure grid weights
        self.grid_columnconfigure(1, weight=3)  # Visualization takes more space
        self.grid_columnconfigure(0, weight=1)  # Controls take less space
        self.grid_rowconfigure(0, weight=1)

        # Initialize visualization
        heatmap_params = {'data_shape': (50, 50), 'cmap': 'gray', 'vmin': 0, 'vmax': 255, 'figsize': (5, 5)}
        self.heatmap = AnimatedHeatmap(self.viz_frame, params=heatmap_params)
        self.heatmap.canvas_widget.pack(expand=True, fill="both", padx=5, pady=5)

        # Create control elements
        self._create_control_elements()

    def _create_control_elements(self):
        """Creates all control elements in the left sidebar"""
        # Dataset Path Frame
        dataset_path_frame = ctk.CTkFrame(self.controls_frame)
        dataset_path_frame.grid(row=1, column=0, sticky="ew", pady=5)
        ctk.CTkLabel(dataset_path_frame, text="Dataset Path:").grid(row=0, column=0, sticky="w", padx=5)
        dataset_path_entry = ctk.CTkEntry(dataset_path_frame, textvariable=self.dataset_path_var, width=250)
        dataset_path_entry.grid(row=0, column=1, sticky="ew", padx=5)
        dataset_path_button = ctk.CTkButton(dataset_path_frame, text="Browse", command=self._browse_dataset_path)
        dataset_path_button.grid(row=0, column=2, sticky="w", padx=5)
        
        # Images Per Class Frame
        images_per_class_frame = ctk.CTkFrame(self.controls_frame)
        images_per_class_frame.grid(row=2, column=0, sticky="ew", pady=5)
        ctk.CTkLabel(images_per_class_frame, text="Images/Class:").grid(row=0, column=0, sticky="w", padx=5)
        images_per_class_entry = ctk.CTkEntry(images_per_class_frame, textvariable=self.images_per_class_var, width=80)
        images_per_class_entry.grid(row=0, column=1, sticky="w", padx=5)
        
        # Feature Method Frame
        self._create_feature_method_frame(row_num=3)
        
        # Feature Size Frame
        self._create_feature_size_frame(row_num=4)
        
        # Dataset Split Frame
        self._create_dataset_split_frame(row_num=5)
        
        # Action Buttons Frame
        action_buttons_frame = ctk.CTkFrame(self.controls_frame)
        action_buttons_frame.grid(row=6, column=0, sticky="ew", pady=10)
        
        self.load_dataset_button = ctk.CTkButton(action_buttons_frame, text="Load Dataset", command=self._load_dataset)
        self.load_dataset_button.grid(row=0, column=0, sticky="ew", padx=2, pady=5)
        
        self.prepare_data_button = ctk.CTkButton(action_buttons_frame, text="Prepare Data", command=self._prepare_data)
        self.prepare_data_button.grid(row=0, column=1, sticky="ew", padx=2, pady=5)
        
        self.export_features_button = ctk.CTkButton(action_buttons_frame, text="Export Features", command=self._export_features)
        self.export_features_button.grid(row=0, column=2, sticky="ew", padx=2, pady=5)
        
        # Configure grid weights for action buttons
        action_buttons_frame.grid_columnconfigure((0,1,2), weight=1)

    def _create_progress_frame(self):
        """Creates the Progress Bar Frame"""
        self.progress_frame = ctk.CTkFrame(self)
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="")
        self.progress_label.pack(pady=5)
        self.progress_manager = ProgressManager(self.progress_bar, self.progress_label, self.root)

    def _create_feature_method_frame(self, row_num):
        """Creates the feature method selection frame"""
        feature_method_frame = ctk.CTkFrame(self.controls_frame)
        feature_method_frame.grid(row=row_num, column=0, sticky="ew", pady=5)
        ctk.CTkLabel(feature_method_frame, text="Feature Method:").grid(row=0, column=0, padx=5, pady=5)
        
        methods = ['average', 'sum', 'max']
        for i, method in enumerate(methods):
            rb = ctk.CTkRadioButton(feature_method_frame, text=method.capitalize(),
                                variable=self.feature_method_var, value=method)
            rb.grid(row=0, column=i+1, padx=10, pady=5)
            ToolTip(rb, f"Select {method.capitalize()} pooling")

    def _create_feature_size_frame(self, row_num):
        """Creates the feature size selection frame"""
        feature_size_frame = ctk.CTkFrame(self.controls_frame)
        feature_size_frame.grid(row=row_num, column=0, sticky="ew", pady=5)
        ctk.CTkLabel(feature_size_frame, text="Feature Size:").grid(row=0, column=0, padx=5, pady=5)
        
        sizes = ['5x5', '10x10', '25x25', '50x50 (No Prep)']
        for i, size in enumerate(sizes):
            rb = ctk.CTkRadioButton(feature_size_frame, text=size,
                                variable=self.feature_size_var, value=size)
            rb.grid(row=0, column=i+1, padx=10, pady=5)
            ToolTip(rb, f"Select feature size: {size}")

    def _create_dataset_split_frame(self, row_num):
        """Creates the dataset split ratio frame"""
        split_frame = ctk.CTkFrame(self.controls_frame)
        split_frame.grid(row=row_num, column=0, sticky="ew", pady=5)
        ctk.CTkLabel(split_frame, text="Dataset Split Ratios:").grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        entries = [
            ("Train Ratio:", self.train_ratio_var, "Train dataset ratio (0.0-1.0)"),
            ("Validation Ratio:", self.val_ratio_var, "Validation dataset ratio (0.0-1.0)"),
            ("Test Ratio:", self.test_ratio_var, "Test dataset ratio (0.0-1.0)")
        ]
        
        for i, (label_text, var, tooltip_text) in enumerate(entries):
            ctk.CTkLabel(split_frame, text=label_text).grid(row=i+1, column=0, sticky="w", padx=5)
            entry = ctk.CTkEntry(split_frame, textvariable=var, width=80)
            entry.grid(row=i+1, column=1, sticky="w", padx=5)
            entry.bind("<FocusOut>", self._validate_and_normalize_ratios)
            ToolTip(entry, tooltip_text)
        
        split_frame.grid_columnconfigure(1, weight=1)
        self._validate_and_normalize_ratios()

    def _validate_and_normalize_ratios(self, event=None):
        """Validates ratio entries and normalizes them to sum to 1.0"""
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
        """Browse for dataset directory"""
        root_dir = filedialog.askdirectory(title="Select Dataset Directory")
        if root_dir:
            self.dataset_path_var.set(root_dir)

    def _load_dataset(self):
        """Load image dataset from selected directory"""
        root_dir = self.dataset_path_var.get()
        if not root_dir:
            messagebox.showerror("Error", "Dataset path is required.")
            return
        
        # Pre-validate directory structure
        train_dir = os.path.join(root_dir, 'train')
        if not os.path.exists(root_dir):
            messagebox.showerror("Error", f"Dataset directory not found: {root_dir}")
            return
        if not os.path.isdir(train_dir):
            messagebox.showerror("Error",
                             "Invalid dataset structure!\n\n"
                             "Expected structure:\n"
                             "selected_directory/\n"
                             "└── train/\n"
                             "    ├── 0/\n"
                             "    ├── 1/\n"
                             "    └── .../")
            return
            
        try:
            max_num_img_per_label = int(self.images_per_class_var.get())
            if max_num_img_per_label <= 0:
                messagebox.showerror("Error", "Images per class must be positive.")
                return
                
            self.progress_manager.start(mode='indeterminate', text="Validating and Loading Dataset...")
            self.dataset.load_dataset(
                root_dir,
                max_num_img_per_label,
                progress_callback=self.progress_manager.update,
                heatmap_callback=self.heatmap.update_heatmap
            )
            self.progress_manager.stop(text="Dataset Loaded")
            messagebox.showinfo("Info", "Dataset loaded and cropped successfully!")
            # Notify training tab to refresh dataset
            self.root.nametowidget(".!notebook.!trainingtab").refresh_dataset()
            
        except ValueError:
            self.progress_manager.stop(text="Load Failed")
            messagebox.showerror("Error", "Invalid Images per Class value.")
        except FileNotFoundError:
            self.progress_manager.stop(text="Load Failed")
            messagebox.showerror("Error", str(e))
        except Exception as e:
            self.progress_manager.stop(text="Load Failed")
            messagebox.showerror("Error",
                             "Dataset Loading Error\n\n"
                             f"Details: {str(e)}\n\n"
                             "Please ensure:\n"
                             "1. The selected directory contains a 'train' folder\n"
                             "2. The 'train' folder contains numbered subdirectories (0,1,2,...)\n"
                             "3. Each numbered directory contains valid image files")

    def _prepare_data(self):
        """Prepare features from the loaded dataset and split dataset"""
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
            # Notify training tab to refresh dataset
            self.root.nametowidget(".!notebook.!trainingtab").refresh_dataset()
            messagebox.showinfo("Info", f"Dataset features prepared successfully using {feature_method.capitalize()} "
                              f"method and feature size {feature_size_str}, and dataset split!")
                              
        except ValueError as e:
            self.progress_manager.stop(text="Preparation Failed")
            messagebox.showerror("Error", str(e))
        except Exception as e:
            self.progress_manager.stop(text="Preparation Failed")
            messagebox.showerror("Error", f"Data preparation error: {e}")

    def _export_features(self):
        """Export prepared features to a CSV file"""
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

    # Methods for training tab access
    def get_dataset(self):
        """Get the dataset instance"""
        return self.dataset

    def get_feature_size(self):
        """Get current feature size tuple"""
        size_str = self.feature_size_var.get()
        if size_str == '5x5':
            return (5, 5)
        elif size_str == '10x10':
            return (10, 10)
        elif size_str == '25x25':
            return (25, 25)
        elif size_str == '50x50 (No Prep)':
            return (50, 50)
        return (5, 5)  # Default