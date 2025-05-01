"""
File Management Module

This module provides utilities for managing file operations,
particularly focused on saving images with automatic file naming.
"""
import os
import time
from PIL import Image

class FileManager:
    """Manages file operations for saving images with automatic naming."""
    
    def __init__(self, filename_var, auto_count, start_number):
        """
        Initialize the FileManager.

        Args:
            filename_var: StringVar containing the base filename (digit or 'unknown')
            auto_count: BooleanVar indicating if auto-incrementing filenames
            start_number: IntVar containing the starting number for auto-increment
        """
        self.filename_var = filename_var
        self.auto_count = auto_count
        self.start_number = start_number
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Ensure the data/processed/dataset directory exists."""
        self.data_dir = os.path.join("data", "processed", "dataset")
        os.makedirs(self.data_dir, exist_ok=True)

    def save_image(self, image):
        """
        Save the image with automatic file naming.

        Args:
            image: PIL Image object to save

        Returns:
            str: Path where the image was saved, or None if save failed
        """
        try:
            # Generate filename with timestamp
            timestamp = int(time.time())
            base_name = self.filename_var.get()
            
            if self.auto_count.get():
                # Auto-incrementing filename
                filename = f"{base_name}_{self.start_number.get()}.png"
                self.start_number.set(self.start_number.get() + 1)
            else:
                # Timestamp-based filename
                filename = f"{base_name}_{timestamp}.png"
            
            # Full path for saving
            save_path = os.path.join(self.data_dir, filename)
            
            # Save the image
            image.save(save_path)
            print(f"Saved image to: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return None