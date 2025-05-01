"""
Preview Module - Handles image preview functionality
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Constants
FINAL_SIZE = 50

class ImagePreviewFrame:
    """Manages the preview of saved images"""
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