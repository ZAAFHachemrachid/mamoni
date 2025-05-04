"""
Canvas Manager Module - Handles drawing canvas operations
"""
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from typing import Callable, Optional, Tuple

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
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.stroke_callback: Optional[Callable[[Image.Image], None]] = None
        
        # Create the canvas
        self.canvas = tk.Canvas(
            parent, width=canvas_size, height=canvas_size,
            bg="white", bd=2, highlightthickness=2, highlightbackground="black"
        )
        
        # Make canvas focusable
        self.canvas.config(takefocus=1)

        # Create PIL image and drawing context
        self.image = Image.new("L", (canvas_size, canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Event bindings
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.canvas.bind("<Button-3>", self.delete_stroke)
        
        # Touch event support
        self.canvas.bind("<<Double-Button-1>>", self.start_stroke)  # Touch tap
        
        # Change highlight color when focused
        self.canvas.bind("<FocusIn>", lambda e: self.canvas.config(highlightbackground="blue"))
        self.canvas.bind("<FocusOut>", lambda e: self.canvas.config(highlightbackground="black"))

    def start_stroke(self, event):
        """Start a new stroke"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        self.canvas.focus_set()
        
    def paint(self, event):
        """Draw on canvas and update internal image"""
        if not self.drawing:
            return
            
        r = self.brush_radius_var.get() if self.brush_radius_var else DEFAULT_BRUSH_RADIUS
        x, y = event.x, event.y
        
        # Draw line between last and current position for smooth strokes
        if self.last_x and self.last_y:
            points = self._interpolate_points((self.last_x, self.last_y), (x, y), r)
            for px, py in points:
                oval_id = self.canvas.create_oval(px - r, py - r, px + r, py + r,
                                               fill="black", outline="black")
                self.stroke_history.append((px - r, py - r, px + r, py + r, r, oval_id))
                self.draw.ellipse([px - r, py - r, px + r, py + r], fill="black")
        
        self.last_x = x
        self.last_y = y
        self.update_bbox()
        
    def end_stroke(self, event):
        """End the current stroke and notify callback if set"""
        self.drawing = False
        self.last_x = None
        self.last_y = None
        if self.stroke_callback:
            self.stroke_callback(self.get_processed_image())
            
    def _interpolate_points(self, p1: Tuple[int, int], p2: Tuple[int, int],
                          step: int = 5) -> list:
        """Interpolate points between two coordinates for smooth lines"""
        x1, y1 = p1
        x2, y2 = p2
        points = []
        
        dx = x2 - x1
        dy = y2 - y1
        distance = max(abs(dx), abs(dy))
        
        if distance == 0:
            return [(x1, y1)]
            
        step_x = dx / (distance / step)
        step_y = dy / (distance / step)
        
        for i in range(int(distance / step)):
            x = x1 + step_x * i
            y = y1 + step_y * i
            points.append((int(x), int(y)))
            
        points.append((x2, y2))
        return points
        

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
            x1, y1, x2, y2, _, oval_id = self.stroke_history.pop()
            self.canvas.delete(oval_id)
            self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
            self.draw = ImageDraw.Draw(self.image)
            for stroke in self.stroke_history:
                x1, y1, x2, y2, _, _ = stroke
                self.draw.ellipse([x1, y1, x2, y2], fill="black")
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