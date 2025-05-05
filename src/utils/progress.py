"""
Progress Management Module

This module provides utilities for managing progress bars and status updates
in the application's user interface.
"""
import tkinter as tk
from enum import Enum, auto
import asyncio

class ProgressState(Enum):
    """States for progress tracking"""
    IDLE = auto()
    LOADING = auto()
    COMPLETED = auto()
    ERROR = auto()

class ProgressManager:
    """Manages progress bar and status updates."""
    def __init__(self, progress_bar, progress_label, root):
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        self.root = root
        self.state = ProgressState.IDLE
        self._current_task = None

    def start(self, mode='indeterminate', text="Processing..."):
        """Start progress indication."""
        self.state = ProgressState.LOADING
        self.progress_bar.configure(mode=mode)
        if mode == 'indeterminate':
            self.progress_bar.start()
        self.progress_label.configure(text=text)
        self.root.update()

    def update(self, value, max_value, text=None):
        """Update progress bar with current value."""
        self.progress_bar.configure(mode='determinate')
        # Convert to 0-1 range for CustomTkinter
        progress = float(value) / float(max_value)
        self.progress_bar.set(progress)
        if text:
            self.progress_label.configure(text=text)
        self.root.update_idletasks()

    def stop(self, text="Done", success=True):
        """Stop progress indication."""
        self.progress_bar.stop()
        self.state = ProgressState.COMPLETED if success else ProgressState.ERROR
        self.progress_label.configure(text=text)
        self.progress_bar.configure(mode='determinate')
        self.progress_bar.set(1.0 if success else 0.0)
        self.root.update()

    def error(self, error_text):
        """Show error state."""
        self.state = ProgressState.ERROR
        self.progress_bar.stop()
        self.progress_label.configure(text=f"Error: {error_text}")
        self.progress_bar.configure(mode='determinate')
        self.progress_bar.set(0.0)
        self.root.update()

    async def track_async_task(self, task, description="Loading..."):
        """Track progress of an async task.
        
        Args:
            task: Coroutine to execute
            description: Status text to display
        
        Returns:
            Result of the task
        """
        self._current_task = task
        self.start(text=description)
        
        try:
            result = await task
            self.stop(text="Completed successfully")
            return result
        except Exception as e:
            self.error(str(e))
            raise

    def is_busy(self):
        """Check if a task is currently in progress."""
        return self.state == ProgressState.LOADING

    def get_state(self):
        """Get current progress state."""
        return self.state