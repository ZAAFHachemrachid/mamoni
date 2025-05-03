"""
UI Tabs Package - Contains tab-specific UI components.
"""
from .data_preparation import DataPreparationTab
from .drawing import DrawingAppTab
from .prediction import PredictionTab
from .training import TrainingTab

__all__ = [
    'DataPreparationTab',
    'DrawingAppTab',
    'PredictionTab',
    'TrainingTab'
]

# Define tab order for the application
TAB_ORDER = [
    DataPreparationTab,  # Data preparation first
    TrainingTab,        # Training depends on data preparation
    PredictionTab       # Prediction depends on training and data
]