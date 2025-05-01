import numpy as np

class NeuralNetwork:
    """Neural network implementation with training and prediction capabilities."""
    
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize weights and biases for all layers."""
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2 / self.layer_sizes[i]))
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))

    def forward(self, X):
        """Forward pass through the network."""
        activations = [X]
        for i in range(len(self.weights)):
            net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                # ReLU activation for hidden layers
                activation = np.maximum(0, net)
            else:
                # Softmax activation for output layer
                exp_net = np.exp(net - np.max(net, axis=1, keepdims=True))
                activation = exp_net / np.sum(exp_net, axis=1, keepdims=True)
            activations.append(activation)
        return activations

    def backward(self, X, y, activations, lr):
        """Backward pass to update weights and biases."""
        n_samples = X.shape[0]
        deltas = []

        # Output layer error
        delta = activations[-1] - y
        deltas.append(delta)

        # Hidden layers error
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(delta, self.weights[i].T)
            delta = delta * (activations[i] > 0)  # ReLU derivative
            deltas.append(delta)
        deltas = deltas[::-1]

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= lr * np.dot(activations[i].T, deltas[i]) / n_samples
            self.biases[i] -= lr * np.sum(deltas[i], axis=0, keepdims=True) / n_samples

    def train_epoch(self, X, y, lr=0.01, batch_size=128):
        """Train the network for one epoch."""
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        
        total_loss = 0
        for start_idx in range(0, X.shape[0], batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Forward pass
            activations = self.forward(X_batch)
            
            # Backward pass
            self.backward(X_batch, y_batch, activations, lr)
            
            # Calculate loss
            log_probs = np.log(activations[-1] + 1e-8)
            total_loss -= np.sum(y_batch * log_probs)
            
        return total_loss / X.shape[0]

    def predict(self, X):
        """Predict class labels for input X."""
        return np.argmax(self.forward(X)[-1], axis=1)

    def save_model(self, filepath):
        """Save model parameters to file."""
        model_params = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'layer_sizes': self.layer_sizes
        }
        np.save(filepath, model_params)

    def load_model(self, filepath):
        """Load model parameters from file."""
        model_params = np.load(filepath, allow_pickle=True).item()
        self.weights = [np.array(w) for w in model_params['weights']]
        self.biases = [np.array(b) for b in model_params['biases']]
        self.layer_sizes = model_params['layer_sizes']