import numpy as np
from src.core.controller import NeuralNetController
from src.data.data_layer import ImageDataset

def test_accuracy_logging():
    """Test accuracy calculation and logging improvements"""
    
    print("\nTesting Accuracy Calculation and Logging\n" + "="*40)
    
    # Create test dataset
    dataset = ImageDataset()
    dataset.features = np.random.randn(100, 25)  # 100 samples, 25 features
    
    # Create labels with known distribution
    true_labels = np.random.randint(0, 3, size=100)  # 3 classes
    dataset.encoded_labels = np.eye(3)[true_labels]  # One-hot encode
    dataset.num_classes = 3
    
    # Split dataset with small sizes for quick testing
    dataset.split_dataset(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # Initialize controller
    controller = NeuralNetController(dataset)
    
    # Create simple model
    controller.create_model([25, 10, 3])  # Input -> Hidden -> Output
    
    print("\nTraining Validation\n" + "-"*40)
    
    # Run single training epoch
    loss, accuracy = controller.train_epoch(learning_rate=0.01, batch_size=10)
    
    print("\nChecking Results\n" + "-"*40)
    print(f"Training Loss: {loss:.4f}")
    print(f"Training Accuracy: {accuracy:.4f}")
    
    print("\nValidation Results\n" + "-"*40)
    val_loss, val_accuracy = controller.validate()
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Verify outputs are reasonable
    assert 0 <= accuracy <= 1, f"Invalid accuracy value: {accuracy}"
    assert 0 <= val_accuracy <= 1, f"Invalid validation accuracy value: {val_accuracy}"
    assert not np.isnan(loss), f"Loss is NaN"
    assert not np.isnan(val_loss), f"Validation loss is NaN"
    
    print("\nAll accuracy and logging validations passed!")

if __name__ == "__main__":
    test_accuracy_logging()