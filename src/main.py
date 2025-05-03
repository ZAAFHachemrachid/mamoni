"""
Application Entry Point

This module serves as the entry point for the digit recognition application.
"""
import customtkinter as ctk
from src.app import MergedApp

def main():
    """
    Main entry point for the application.
    Initializes and runs the GUI application.
    """
    # Set default color theme
    ctk.set_default_color_theme("blue")
    
    # Create and run application
    app = MergedApp()
    app.mainloop()

if __name__ == "__main__":
    main()
