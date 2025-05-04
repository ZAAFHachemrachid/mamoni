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

from src.ui.tabs import TAB_ORDER, DataPreparationTab, TrainingTab, PredictionTab

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

        # Initialize tabs based on TAB_ORDER
        self.tabs = {}
        self.tab_instances = {}
        
        # Create tabs according to defined order
        for tab_class in TAB_ORDER:
            tab_name = tab_class.__name__.replace('Tab', '')
            self.tabs[tab_name] = self.tabview.add(tab_name)
            self.tabs[tab_name].grid_columnconfigure(0, weight=1)
            self.tabs[tab_name].grid_rowconfigure(0, weight=1)

        # Initialize tabs in the correct order with proper dependencies
        for tab_class in TAB_ORDER:
            tab_name = tab_class.__name__.replace('Tab', '')
            
            # Data Preparation Tab - initialized first
            if tab_class == DataPreparationTab:
                data_prep_tab = DataPreparationTab(self.tabs[tab_name])
                data_prep_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
                self.tab_instances[tab_name] = data_prep_tab
            
            # Training Tab - depends on data_prep_tab
            elif tab_class == TrainingTab:
                training_tab = TrainingTab(self.tabs[tab_name], data_prep_tab)
                training_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
                self.tab_instances[tab_name] = training_tab
            
            # Prediction Tab - depends on controller and dataset
            elif tab_class == PredictionTab:
                prediction_tab = PredictionTab(
                    self.tabs[tab_name],
                    training_tab.controller,
                    data_prep_tab.get_dataset()
                )
                prediction_tab.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
                self.tab_instances[tab_name] = prediction_tab

        # Configure tab change callback
        self.tabview.configure(command=self.on_tab_changed)
        
        # Set initial tab
        self.tabview.set("DataPreparation")

        # Setup keyboard shortcuts
        self.setup_keyboard_shortcuts()

    def setup_keyboard_shortcuts(self):
        """Setup global keyboard shortcuts."""
        # No shortcuts needed for now
        self.drawing_shortcuts = {}

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
        for char in self.drawing_shortcuts:
            self.unbind(f'<{char}>')

        # Rebind shortcuts if on Drawing tab
        if current_tab == 'Drawing':
            for char, command in self.drawing_shortcuts.items():
                self.bind(f'<{char}>', lambda event, cmd=command: cmd())