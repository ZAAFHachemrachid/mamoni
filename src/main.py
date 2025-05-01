"""
Application Entry Point

This module serves as the entry point for the digit recognition application.
"""
import customtkinter as ctk
from app import MergedApp

if __name__ == "__main__":
    # Set default color theme
    ctk.set_default_color_theme("blue")
    
    # Create and run application
    app = MergedApp()
    app.mainloop()
