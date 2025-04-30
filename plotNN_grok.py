import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class NeuralNetworkPlotter:
    """Handles neural network plotting and visualization logic."""
    
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        
    def plot(self, layers, input_values=None, output_values=None):
        """
        Plot a neural network with the given layer sizes.
        
        Args:
            layers (list): List of integers representing the number of neurons in each layer
            input_values (list, optional): Values for input layer nodes
            output_values (list, optional): Values for output layer nodes
        """
        self.ax.clear()
        
        n_layers = len(layers)
        x_positions = np.linspace(0, 1, n_layers)
        y_positions = []
        
        # Calculate y-positions for all nodes
        for n_neurons in layers:
            layer_positions = np.linspace(-(n_neurons-1)/2, (n_neurons-1)/2, n_neurons)
            y_positions.append(layer_positions)
        
        # Determine visible nodes for each layer (show top/bottom 10 for large layers)
        visible_indices = self._get_visible_indices(layers)
        
        # Draw nodes
        self._draw_nodes(n_layers, x_positions, y_positions, visible_indices, 
                        input_values, output_values)
        
        # Draw connections
        self._draw_connections(n_layers, x_positions, y_positions, visible_indices)
        
        # Configure plot appearance
        self._configure_plot(x_positions, y_positions, layers)
        
    def _get_visible_indices(self, layers):
        """Determine which nodes should be visible in each layer."""
        visible_indices = []
        for n_neurons in layers:
            if n_neurons <= 64:
                visible_indices.append(list(range(n_neurons)))
            else:
                visible_indices.append(list(range(10)) + list(range(n_neurons-10, n_neurons)))
        return visible_indices
        
    def _draw_nodes(self, n_layers, x_positions, y_positions, visible_indices, 
                input_values, output_values):
        """Draw the nodes of the neural network with separated value displays."""
        # Node offset for better separation between nodes and values
        input_offset = 0.05
        output_offset = 0.05
        
        for i in range(n_layers):
            y_pos = y_positions[i]
            
            # For output layer, find max among visible nodes
            max_output = None
            if i == n_layers - 1 and output_values is not None:
                visible_output_values = [output_values[j] for j in visible_indices[i]]
                max_output = max(visible_output_values)
            
            for j in visible_indices[i]:
                # Determine node color
                current_color = 'skyblue'
                if i == n_layers - 1 and output_values is not None:
                    if output_values[j] == max_output:
                        current_color = 'gold'  # Highlight max among visible output nodes
                
                # Plot the nodes separately from their values
                self.ax.plot(x_positions[i], y_pos[j], 'o', color=current_color, markersize=8)
                
                # Add separate value displays for input and output layers
                if i == 0 and input_values is not None:
                    value = input_values[j]
                    # Draw the input value in a separate position
                    self.ax.plot(x_positions[i] - input_offset, y_pos[j], 's', color='lightgreen', markersize=6)
                    self.ax.text(x_positions[i] - input_offset, y_pos[j], f"{value:.2f}",
                            ha='center', va='center', color='black', fontsize=8)
                    # Draw a light connecting line
                    self.ax.plot([x_positions[i] - input_offset, x_positions[i]], 
                            [y_pos[j], y_pos[j]], 'k-', linewidth=0.5, alpha=0.3)
                
                elif i == n_layers - 1 and output_values is not None:
                    value = output_values[j]
                    # Draw the output value in a separate position
                    self.ax.plot(x_positions[i] + output_offset, y_pos[j], 's', color='lightsalmon', markersize=6)
                    self.ax.text(x_positions[i] + output_offset, y_pos[j], f"{value:.2f}",
                            ha='center', va='center', color='black', fontsize=8)
                    # Draw a light connecting line
                    self.ax.plot([x_positions[i], x_positions[i] + output_offset], 
                            [y_pos[j], y_pos[j]], 'k-', linewidth=0.5, alpha=0.3)
        
        # Add vertical ellipsis for layers with > 30 nodes
        for i in range(n_layers):
            if len(y_positions[i]) > 30:
                self.ax.text(x_positions[i], 0, r'$\vdots$', va='center', ha='center', fontsize=12)

    def _draw_connections(self, n_layers, x_positions, y_positions, visible_indices):
        """Draw connections between nodes in adjacent layers."""
        for i in range(n_layers - 1):
            for j in visible_indices[i]:
                for k in visible_indices[i+1]:
                    self.ax.plot([x_positions[i], x_positions[i+1]],
                            [y_positions[i][j], y_positions[i+1][k]],
                            'k-', linewidth=0.5, alpha=0.6)
    
    def _configure_plot(self, x_positions, y_positions, layers):
        """Configure plot appearance, labels, and limits with space for separated values."""
        # Expand x limits to accommodate the separated input/output values
        self.ax.set_xlim(-0.2, 1.2)
        y_min = min(np.min(y_pos) for y_pos in y_positions)
        y_max = max(np.max(y_pos) for y_pos in y_positions)
        self.ax.set_ylim(y_min - 1, y_max + 1)
        
        self.ax.set_xticks(x_positions)
        self.ax.set_xticklabels([f"Layer {i+1} ({n})" for i, n in enumerate(layers)])
        self.ax.set_yticks([])
        
        self.ax.set_title("Feedforward Neural Network Visualization\n"
                    "(Input nodes: blue | Input values: green | Output nodes: skyblue/gold | Output values: salmon)")
        
class NetworkConfigPanel(ttk.Frame):
    """Panel for configuring the neural network parameters."""
    
    def __init__(self, master, update_callback):
        super().__init__(master)
        self.update_callback = update_callback
        
        # Create UI components
        ttk.Label(self, text="Layer sizes (comma-separated):").pack(side=tk.LEFT)
        
        self.layer_entry = ttk.Entry(self, width=20)
        self.layer_entry.pack(side=tk.LEFT, padx=5)
        self.layer_entry.insert(0, "2,3,4")  # Default example
        
        ttk.Button(self, text="Plot", command=self._on_plot_clicked).pack(side=tk.LEFT, padx=5)
    
    def _on_plot_clicked(self):
        """Handle plot button click event."""
        try:
            layers = self._parse_layers()
            self.update_callback(layers)
        except Exception as e:
            messagebox.showerror("Input Error", 
                                f"Error: {str(e)}\n\n"
                                "Example format: '2,3,4' (minimum 2 layers)")
    
    def _parse_layers(self):
        """Parse layer configuration from the input field."""
        layer_str = self.layer_entry.get().strip()
        layers = list(map(int, layer_str.split(',')))
        
        if len(layers) < 2:
            raise ValueError("Need at least 2 layers")
            
        return layers


class NetworkDataGenerator:
    """Generates sample data for neural network inputs and outputs."""
    
    @staticmethod
    def generate_random_values(layers):
        """Generate random values for input and output layers."""
        input_size = layers[0]
        output_size = layers[-1]
        
        input_values = np.random.uniform(0, 1, input_size).round(2).tolist()
        output_values = np.random.uniform(0, 1, output_size).round(2).tolist()
        
        return input_values, output_values


class NeuralNetworkVisualizer(tk.Tk):
    """Main application for neural network visualization."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Neural Network Visualizer with Node Values")
        self.geometry("800x600")
        
        # Create plotting components
        self.fig = Figure(figsize=(4, 3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create neural network plotter
        self.plotter = NeuralNetworkPlotter(self.fig, self.ax)
        
        # Create network configuration panel
        self.config_panel = NetworkConfigPanel(self, self.update_plot)
        self.config_panel.pack(side=tk.BOTTOM, pady=10)
    
    def update_plot(self, layers):
        """Update the neural network plot with the given layer configuration."""
        # Generate random values for network nodes
        input_values, output_values = NetworkDataGenerator.generate_random_values(layers)
        
        # Plot the network
        self.plotter.plot(layers, input_values, output_values)
        self.canvas.draw()


def main():
    app = NeuralNetworkVisualizer()
    app.mainloop()


if __name__ == "__main__":
    main()