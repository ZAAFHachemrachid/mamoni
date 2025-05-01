"""
Progress Management Module

This module provides utilities for managing progress bars and status updates
in the application's user interface.
"""
import tkinter as tk

class ProgressManager:
    """Manages progress bar and status updates."""
    def __init__(self, progress_bar, progress_label, root):
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        self.root = root

    def start(self, mode='indeterminate', text="Processing..."):
        """Start progress indication."""
        self.progress_bar['mode'] = mode
        if mode == 'indeterminate':
            self.progress_bar.start()
        self.progress_label.config(text=text)
        self.root.update()

    def update(self, value, max_value, text=None):
        """Update progress bar with current value."""
        self.progress_bar['mode'] = 'determinate'
        self.progress_bar['maximum'] = max_value
        self.progress_bar['value'] = value
        if text:
            self.progress_label.config(text=text)
        self.root.update_idletasks()

    def stop(self, text="Done"):
        """Stop progress indication."""
        self.progress_bar.stop()
        self.progress_label.config(text=text)
        self.progress_bar['mode'] = 'determinate'
        self.root.update()