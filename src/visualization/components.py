import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ======================================
# Visualization Components
# ======================================

class TrainingMetrics:
    """Plots training metrics (loss and accuracy) over epochs."""

    def __init__(self, parent, params=None):
        self.parent = parent
        self.loss = []
        self.accuracy = []
        self.val_loss = [] # Add validation loss
        self.val_accuracy = [] # Add validation accuracy
        self.epochs = []

        # Default parameters
        default_params = {
            'figsize': (3, 2),
            'dpi': 100,
            'xlim': (0, 10),
            'ylim': (0, 1),
            'interval': 100,
        }

        # Update default parameters with provided parameters
        params = params or {}
        config = default_params.copy()
        config.update(params)

        self.figsize = config['figsize']
        self.dpi = config['dpi']
        self.xlim = config['xlim']
        self.ylim = config['ylim']
        self.interval = config['interval']

        # Set up figure and axes
        self.figure = Figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.figure.add_subplot(111)
        self.loss_line, = self.ax.plot([], [], 'b-', label='Train Loss') # Label changed
        self.acc_line, = self.ax.plot([], [], 'r-', label='Train Accuracy') # Label changed
        self.val_loss_line, = self.ax.plot([], [], 'm--', label='Val Loss') # Validation loss line
        self.val_acc_line, = self.ax.plot([], [], 'g--', label='Val Accuracy') # Validation accuracy line

        self.ax.legend()
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_xlabel("Epochs")
        self.ax.set_ylabel("Metrics")
        self.ax.set_title("Training Progress")

        # Set up canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent)
        self.canvas_widget = self.canvas.get_tk_widget()

    def reset(self):
        """Reset plot data and redraw."""
        self.epochs = []
        self.loss = []
        self.accuracy = []
        self.val_loss = [] # Reset validation loss
        self.val_accuracy = [] # Reset validation accuracy
        self.loss_line.set_data([], [])
        self.acc_line.set_data([], [])
        self.val_loss_line.set_data([], []) # Reset validation lines
        self.val_acc_line.set_data([], []) # Reset validation lines
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.canvas.draw_idle()

    def update_plot(self, epoch, loss, accuracy, val_loss=None, val_accuracy=None): # Added validation metrics
        """Update plot with new training metrics."""
        print(f"Updating plot: Epoch={epoch}, Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        self.epochs.append(epoch)
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.loss_line.set_data(self.epochs, self.loss)
        self.acc_line.set_data(self.epochs, self.accuracy)

        if val_loss is not None and val_accuracy is not None: # Update validation lines if data is provided
            self.val_loss.append(val_loss)
            self.val_accuracy.append(val_accuracy)
            self.val_loss_line.set_data(self.epochs, self.val_loss)
            self.val_acc_line.set_data(self.epochs, self.val_accuracy)

        # Dynamically adjust x-axis limit
        if epoch > self.ax.get_xlim()[1]:
            self.ax.set_xlim(0, epoch + max(1, int(0.1 * epoch)))

        # Re-draw the canvas
        self.canvas.draw()


class AnimatedHeatmap:
    """Displays image data as a heatmap for visualization."""

    def __init__(self, parent, params=None):
        self.parent = parent
        config = {
            'data_shape': (50, 50),
            'figsize': (5, 4),
            'dpi': 100,
            'cmap': 'gray',
            'vmin': 0,
            'vmax': 255,
            'interval': 100,
            **(params or {})
        }

        for key, value in config.items():
            setattr(self, key, value)

        self.data = np.zeros(self.data_shape)
        self.figure = Figure(figsize=self.figsize, dpi=self.dpi)
        self.ax = self.figure.add_subplot(111)
        self.im = self.ax.imshow(self.data, cmap=self.cmap,
                                vmin=self.vmin, vmax=self.vmax,
                                interpolation='nearest')
        self.figure.colorbar(self.im)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.parent)
        self.canvas_widget = self.canvas.get_tk_widget()

    def update_heatmap(self, new_data):
        """Update heatmap with new image data."""
        if new_data is not None:
            if new_data.shape != self.data_shape:
                new_data = np.array(Image.fromarray(new_data.astype('uint8')).resize(
                                    self.data_shape, Image.LANCZOS))
            self.data = new_data
        else:
            self.data = np.zeros(self.data_shape)

        self.im.set_array(self.data)
        self.canvas.draw_idle()

