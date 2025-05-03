"""
Preview Module - Handles image preview functionality
"""
import customtkinter as ctk
from PIL import Image, ImageTk

# Constants
FINAL_SIZE = 50

class ImagePreviewFrame(ctk.CTkFrame):
    """Manages the preview of saved images"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Add title label to replace LabelFrame functionality
        self.title_label = ctk.CTkLabel(
            self,
            text="Last Saved Image",
            font=ctk.CTkFont(weight="bold")
        )
        self.title_label.pack(padx=5, pady=(5, 0))
        
        # Preview label needs to remain tk.Label for PhotoImage compatibility
        # But we'll style it to match customtkinter theme
        self.preview_label = ctk.CTkLabel(
            self,
            text=""  # Empty text as we'll use it for images
        )
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
        self.preview_label._image = self.photo_image  # Store reference to prevent garbage collection
        self.preview_label.configure(image=self.photo_image)