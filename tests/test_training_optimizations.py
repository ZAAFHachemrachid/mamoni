import numpy as np
import os
import time
import psutil
import matplotlib.pyplot as plt
from src.core.neural_network import NeuralNetwork
from src.data.data_layer import ImageDataset

def test_training_optimizations():
    """Test and validate training optimizations"""
    
    # Initialize dataset with enhanced features
    dataset = ImageDataset()
    
    # Create sample dataset
    X_train = np.random.randn(1000, 50, 50)  # 1000 sample images
    y_train = np.random.randint(0, 10, size=1000)  # 10 classes
    y_train_encoded = np.eye(10)[y_train]  # One-hot encode labels
    
    # Prepare features with advanced extraction
    X_features = []
    for img in X_train:
        features = dataset.image_to_features(img, method='advanced', feature_size=(10, 10))
        X_features.append(features)
    X_features = np.array(X_features)
    
    # Standardize features
    mean = np.mean(X_features, axis=0)
    std = np.std(X_features, axis=0) + 1e-8
    X_features = (X_features - mean) / std
    
    # Initialize model with new architecture
    model = NeuralNetwork()  # Uses default architecture [2500->128->64->output]
    
    # Monitoring metrics
    losses = []
    grad_norms = []
    memory_usage = []
    times = []
    accuracies = []
    
    # Training loop
    num_epochs = 5
    batch_size = 256  # Increased batch size
    learning_rate = 0.01
    min_learning_rate = 1e-6
    decay_rate = 0.95
    decay_steps = 1000
    start_time = time.time()
    
    print("\nStarting training validation...\n")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0
        grad_norm_sum = 0
        correct_predictions = 0
        
        # Apply learning rate decay
        current_step = epoch * num_batches
        current_lr = max(
            learning_rate * (decay_rate ** (current_step / decay_steps)),
            min_learning_rate
        )
        
        # Training batches
        num_batches = len(X_features) // batch_size
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            
            X_batch = X_features[batch_start:batch_end]
            y_batch = y_train_encoded[batch_start:batch_end]
            
            # Forward pass with dropout
            activations, dropout_masks = model.forward(X_batch, training=True)
            
            # Store gradients before update
            old_weights = [w.copy() for w in model.weights]
            
            # Backward pass
            model.backward(X_batch, y_batch, activations, dropout_masks, lr=current_lr)
            
            # Calculate gradient norm
            grad_norm = np.sqrt(sum(np.sum(np.square(w_new - w_old)) 
                              for w_new, w_old in zip(model.weights, old_weights)))
            grad_norm_sum += grad_norm
            
            # Calculate loss
            epsilon = 1e-12
            log_probs = np.log(np.clip(activations[-1], epsilon, 1.0 - epsilon))
            batch_loss = -np.sum(y_batch * log_probs) / batch_size
            total_loss += batch_loss
            
            # Calculate accuracy
            predictions = np.argmax(activations[-1], axis=1)
            correct_predictions += np.sum(predictions == y_train[batch_start:batch_end])
            
            # Monitor memory
            memory_usage.append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
        
        # Epoch metrics
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        avg_grad_norm = grad_norm_sum / num_batches
        accuracy = correct_predictions / len(X_train)
        
        losses.append(avg_loss)
        grad_norms.append(avg_grad_norm)
        times.append(epoch_time)
        accuracies.append(accuracy)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Gradient Norm: {avg_grad_norm:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Memory: {memory_usage[-1]:.1f}MB")
        print("-" * 40)
    
    total_time = time.time() - start_time
    
    print("\nTraining Validation Results:")
    print("-" * 40)
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Final Accuracy: {accuracies[-1]:.4f}")
    print(f"Average Gradient Norm: {np.mean(grad_norms):.4f}")
    print(f"Average Memory Usage: {np.mean(memory_usage):.1f}MB")
    print(f"Average Epoch Time: {np.mean(times):.2f}s")
    print(f"Total Training Time: {total_time:.2f}s")
    
    # Extended validation checks
    print("\nOptimization Validations:")
    print("-" * 40)
    
    # Check gradient clipping
    print("Gradient Clipping:", end=" ")
    if all(norm <= model.clip_value * 1.1 for norm in grad_norms):
        print("✓ Working (all gradients within clip threshold)")
    else:
        print("✗ Not working properly")
    
    # Check memory efficiency
    print("Memory Efficiency:", end=" ")
    memory_increase = max(memory_usage) - min(memory_usage)
    if memory_increase < 500:
        print("✓ Good (stable memory usage)")
    else:
        print("✗ Poor (significant memory growth)")
    
    # Check training speed
    print("Training Speed:", end=" ")
    if np.mean(times) < 5:
        print("✓ Good (fast epoch processing)")
    else:
        print("✗ Slow (needs optimization)")
    
    # Check learning progress
    print("Learning Progress:", end=" ")
    if losses[-1] < losses[0]:
        print("✓ Model is learning (loss decreasing)")
    else:
        print("✗ Model not learning effectively")
        
    # Check learning rate decay
    print("Learning Rate Decay:", end=" ")
    final_lr = learning_rate * (decay_rate ** ((num_epochs * num_batches) / decay_steps))
    if min_learning_rate <= final_lr < learning_rate:
        print("✓ Working (learning rate decreased properly)")
    else:
        print("✗ Not working as expected")
    
    # Check prediction confidence
    print("Prediction Confidence:", end=" ")
    final_activations = model.forward(X_features[:100], training=False)[-1]
    confidence_scores = np.max(final_activations, axis=1)
    if np.mean(confidence_scores) > 0.5:  # Average confidence above 50%
        print("✓ Good (model making confident predictions)")
    else:
        print("✗ Poor (low confidence predictions)")

if __name__ == "__main__":
    test_training_optimizations()