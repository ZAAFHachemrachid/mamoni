"""
Main Application Module

This module contains the main application class that integrates
all components of the digit recognition application.
"""
import tkinter as tk
from tkinter import ttk

from ui.tabs.drawing import DrawingAppTab
from ui.tabs.training import NeuralNetworkAppTab

class MergedApp(tk.Tk):
    """Main application class that manages the tab-based interface."""
    
    def __init__(self):
        super().__init__()
        self.title("Digit Recognition Application")

        # Configure main grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Create Neural Network App Tab
        self.nn_tab_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.nn_tab_frame, text='Neural Network Trainer')
        self.nn_app_tab = NeuralNetworkAppTab(self.nn_tab_frame)
        self.nn_tab_frame.grid_columnconfigure(0, weight=1)
        self.nn_tab_frame.grid_rowconfigure(0, weight=1)
        self.nn_app_tab.grid(row=0, column=0, sticky="nsew")

        # Create Drawing App Tab
        self.drawing_tab_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.drawing_tab_frame, text='Drawing App')
        self.drawing_app_tab = DrawingAppTab(self.drawing_tab_frame, self.nn_app_tab.controller, self.nn_app_tab.dataset)
        self.drawing_tab_frame.grid_columnconfigure(0, weight=1)
        self.drawing_tab_frame.grid_rowconfigure(0, weight=1)

        # Setup keyboard shortcuts for Drawing Tab
        for char, command in {
            's': self.drawing_app_tab.save_image,
            'r': self.drawing_app_tab.reset_canvas,
            'd': self.drawing_app_tab.undo
        }.items():
            self.bind(f'<{char}>', lambda event, cmd=command: cmd())
            
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def on_tab_changed(self, event):
        """Handle tab change event to update keyboard shortcuts."""
        selected_tab = self.notebook.select()
        tab_text = self.notebook.tab(selected_tab, "text")

        # Unbind existing shortcuts
        for char in ['s', 'r', 'd']:
            self.unbind(f'<{char}>')

        if tab_text == 'Drawing App':
            for char, command in {
                's': self.drawing_app_tab.save_image,
                'r': self.drawing_app_tab.reset_canvas,
                'd': self.drawing_app_tab.undo
            }.items():
                self.bind(f'<{char}>', lambda event, cmd=command: cmd())