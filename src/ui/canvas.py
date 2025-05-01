"""
Canvas Manager Module - Handles drawing canvas operations
"""
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

# Constants
DEFAULT_CANVAS_SIZE = 400
DEFAULT_BRUSH_RADIUS = 15
FINAL_SIZE = 50

class CanvasManager:
    """Manages the drawing canvas and its operations"""
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
            _, _, _, _, _, oval_id = self.stroke_history.pop()
            self.canvas.delete(oval_id)
            self.redraw_image()
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