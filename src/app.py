"""
Main Application Module

This module contains the main application class that integrates
all components of the digit recognition application.
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import json
import os

from ui.tabs.drawing import DrawingAppTab
from ui.tabs.training import NeuralNetworkAppTab

class MergedApp(ctk.CTk):
    """Main application class that manages the tab-based interface."""
    
    def __init__(self):
        super().__init__()
        
        # Load saved theme preference
        self.theme_file = "theme_preference.json"
        self.load_theme_preference()
        
        # Configure window
        self.title("Digit Recognition Application")
        self.geometry("1200x800")
        self.minsize(800, 600)
        
        # Configure main grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Create theme toggle button
        self.theme_button = ctk.CTkButton(
            self,
            text="Toggle Theme",
            command=self.toggle_theme,
            width=120
        )
        self.theme_button.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ne")

        # Create tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Create Neural Network Tab
        self.nn_tab = self.tabview.add("Neural Network Trainer")
        self.nn_tab.grid_columnconfigure(0, weight=1)
        self.nn_tab.grid_rowconfigure(0, weight=1)
        self.nn_app_tab = NeuralNetworkAppTab(self.nn_tab)
        self.nn_app_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Create Drawing Tab
        self.drawing_tab = self.tabview.add("Drawing App")
        self.drawing_tab.grid_columnconfigure(0, weight=1)
        self.drawing_tab.grid_rowconfigure(0, weight=1)
        self.drawing_app_tab = DrawingAppTab(self.drawing_tab, self.nn_app_tab.controller, self.nn_app_tab.dataset)
        self.drawing_app_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Setup keyboard shortcuts for Drawing Tab
        for char, command in {
            's': self.drawing_app_tab.save_image,
            'r': self.drawing_app_tab.reset_canvas,
            'd': self.drawing_app_tab.undo
        }.items():
            self.bind(f'<{char}>', lambda event, cmd=command: cmd())
        
        # Configure tab change callback
        self.tabview.configure(command=self.on_tab_changed)
        
        # Set initial tab
        self.tabview.set("Neural Network Trainer")

    def load_theme_preference(self):
        """Load saved theme preference or use system default."""
        try:
            if os.path.exists(self.theme_file):
                with open(self.theme_file, 'r') as f:
                    preference = json.load(f)
                    ctk.set_appearance_mode(preference.get('theme', 'system'))
            else:
                ctk.set_appearance_mode("system")
        except Exception:
            ctk.set_appearance_mode("system")

    def save_theme_preference(self, theme):
        """Save current theme preference."""
        try:
            with open(self.theme_file, 'w') as f:
                json.dump({'theme': theme}, f)
        except Exception:
            pass

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        current = ctk.get_appearance_mode()
        new_theme = "light" if current == "dark" else "dark"
        ctk.set_appearance_mode(new_theme)
        self.save_theme_preference(new_theme)

    def on_tab_changed(self):
        """Handle tab change event to update keyboard shortcuts."""
        current_tab = self.tabview.get()

        # Unbind existing shortcuts
        for char in ['s', 'r', 'd']:
            self.unbind(f'<{char}>')

        if current_tab == 'Drawing App':
            for char, command in {
                's': self.drawing_app_tab.save_image,
                'r': self.drawing_app_tab.reset_canvas,
                'd': self.drawing_app_tab.undo
            }.items():
                self.bind(f'<{char}>', lambda event, cmd=command: cmd())