import numpy as np
import json

# ======================================
# Model Layer
# ======================================


# ======================================
# Model Layer (Already included in your provided code)
# ======================================
class NeuralNetwork:
    """Neural network implementation with training and prediction capabilities."""
    # ... (NeuralNetwork class code as you provided) ...
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize weights and biases using Xavier initialization."""
        for i in range(len(self.layer_sizes)-1):
            scale = np.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i+1]))
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * scale)
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))

    def sigmoid(self, x):
        """Sigmoid activation function with clipping to prevent overflow."""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function for backpropagation."""
        return x * (1 - x)

    def forward(self, X):
        """Forward pass through the network."""
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights)-1:  # Output layer: softmax
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                activations.append(exp_z / exp_z.sum(axis=1, keepdims=True))
            else:  # Hidden layers: sigmoid
                activations.append(self.sigmoid(z))
        return activations

    def backward(self, X, y, activations, lr):
        """Backward pass to update weights and biases."""
        m = X.shape[0]
        deltas = []

        # Output layer delta (softmax cross-entropy derivative)
        delta = (activations[-1] - y.reshape(y.shape[0], -1))
        deltas.append(delta)

        # Hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            delta = np.dot(deltas[-1], self.weights[i+1].T) * self.sigmoid_derivative(activations[i+1])
            deltas.append(delta)
        deltas.reverse()

        # Update parameters
        for i in range(len(self.weights)):
            grad_w = np.dot(activations[i].T, deltas[i]) / m
            grad_b = np.mean(deltas[i], axis=0, keepdims=True)
            self.weights[i] -= lr * grad_w
            self.biases[i] -= lr * grad_b

        # Calculate cross-entropy loss
        log_probs = np.log(activations[-1] + 1e-8)
        return -np.sum(y.reshape(y.shape[0], -1) * log_probs) / m

    def train_epoch(self, X, y, lr=0.01, batch_size=128):
        """Train the model for one epoch using mini-batch gradient descent."""
        indices = np.random.permutation(X.shape[0])
        total_loss = 0

        for i in range(0, X.shape[0], batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch, y_batch = X[batch_idx], y[batch_idx]

            activations = self.forward(X_batch)
            loss = self.backward(X_batch, y_batch, activations, lr)
            total_loss += loss * X_batch.shape[0]

        avg_loss = total_loss / X.shape[0]
        return avg_loss

    def predict(self, X):
        """Predict class labels for input data."""
        return np.argmax(self.forward(X)[-1], axis=1)

    def save_model(self, filepath):
        """Saves the model parameters to a JSON file."""
        model_data = {
            "layer_sizes": [int(size) for size in self.layer_sizes], # Convert layer_sizes to list of int
            "weights": [[[float(val) for val in row] for row in w.tolist()] for w in self.weights], # Explicit float conversion
            "biases": [[[float(val) for val in row] for row in b.tolist()] for b in self.biases]   # Explicit float conversion
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=4)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model parameters from a JSON file."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        self.layer_sizes = model_data["layer_sizes"]
        self.weights = [np.array(w) for w in model_data["weights"]]
        self.biases = [np.array(b) for b in model_data["biases"]]
        print(f"Model loaded from {filepath}")