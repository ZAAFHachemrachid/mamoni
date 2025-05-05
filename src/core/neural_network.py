import numpy as np

class NeuralNetwork:
    """Neural network implementation with training and prediction capabilities."""
    
    def __init__(self, layer_sizes=None):
        """Initialize neural network with optional layer sizes.
        
        Default architecture: [2500->128->64->output]
        """
        self.layer_sizes = layer_sizes or [2500, 128, 64, 10]
        self.dropout_rate = 0.3
        self.clip_value = 5.0
        self._initialize_parameters()
        self._initialize_batch_norm()

    def _initialize_parameters(self):
        """Initialize weights and biases using He initialization."""
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            # He initialization
            scale = np.sqrt(2. / self.layer_sizes[i])
            self.weights.append(np.random.normal(0, scale, (self.layer_sizes[i], self.layer_sizes[i + 1])))
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))

    def _initialize_batch_norm(self):
        """Initialize batch normalization parameters."""
        self.bn_params = []
        for i in range(len(self.layer_sizes) - 2):  # No batch norm on output layer
            self.bn_params.append({
                'gamma': np.ones((1, self.layer_sizes[i + 1])),
                'beta': np.zeros((1, self.layer_sizes[i + 1])),
                'running_mean': np.zeros((1, self.layer_sizes[i + 1])),
                'running_var': np.ones((1, self.layer_sizes[i + 1])),
                'momentum': 0.9
            })

    def forward(self, X, training=False):
        """Forward pass through the network with dropout and batch normalization."""
        activations = [X]
        dropout_masks = []
        
        for i in range(len(self.weights)):
            net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            
            if i < len(self.weights) - 1:  # Hidden layers
                # Apply batch normalization
                if training:
                    batch_mean = np.mean(net, axis=0, keepdims=True)
                    batch_var = np.var(net, axis=0, keepdims=True) + 1e-5
                    net_norm = (net - batch_mean) / np.sqrt(batch_var)
                    
                    # Update running statistics
                    momentum = self.bn_params[i]['momentum']
                    self.bn_params[i]['running_mean'] = (momentum * self.bn_params[i]['running_mean'] +
                                                       (1 - momentum) * batch_mean)
                    self.bn_params[i]['running_var'] = (momentum * self.bn_params[i]['running_var'] +
                                                      (1 - momentum) * batch_var)
                else:
                    net_norm = ((net - self.bn_params[i]['running_mean']) /
                              np.sqrt(self.bn_params[i]['running_var'] + 1e-5))
                
                # Scale and shift
                net = (self.bn_params[i]['gamma'] * net_norm + self.bn_params[i]['beta'])
                
                # ReLU activation
                activation = np.maximum(0, net)
                
                # Apply dropout during training
                if training:
                    mask = (np.random.rand(*activation.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                    activation = activation * mask
                    dropout_masks.append(mask)
                else:
                    dropout_masks.append(None)
            else:  # Output layer
                # Softmax activation
                exp_net = np.exp(net - np.max(net, axis=1, keepdims=True))
                activation = exp_net / np.sum(exp_net, axis=1, keepdims=True)
                
            activations.append(activation)
            
        if training:
            return activations, dropout_masks
        return activations

    def backward(self, X, y, activations, dropout_masks, lr):
        """Backward pass with gradient clipping."""
        n_samples = X.shape[0]
        deltas = []
        bn_grads = []

        # Output layer error
        delta = activations[-1] - y
        deltas.append(delta)

        # Hidden layers error
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(delta, self.weights[i].T)
            
            if i > 0:  # Skip input layer
                # Batch norm backward pass
                bn_grad = self._batch_norm_backward(delta, activations[i], self.bn_params[i-1])
                bn_grads.insert(0, bn_grad)
                
                # ReLU derivative
                delta = delta * (activations[i] > 0)
                
                # Apply dropout mask
                if dropout_masks[i-1] is not None:
                    delta = delta * dropout_masks[i-1]
                    
            deltas.append(delta)
        deltas = deltas[::-1]
        
        # Update weights and biases with gradient clipping
        for i in range(len(self.weights)):
            # Calculate gradients
            weight_grad = np.dot(activations[i].T, deltas[i]) / n_samples
            bias_grad = np.sum(deltas[i], axis=0, keepdims=True) / n_samples
            
            # Clip gradients
            weight_grad = np.clip(weight_grad, -self.clip_value, self.clip_value)
            bias_grad = np.clip(bias_grad, -self.clip_value, self.clip_value)
            
            # Apply updates
            self.weights[i] -= lr * weight_grad
            self.biases[i] -= lr * bias_grad
            
            # Update batch norm parameters for hidden layers
            if i < len(self.weights) - 1:
                self.bn_params[i]['gamma'] -= lr * bn_grads[i]['gamma']
                self.bn_params[i]['beta'] -= lr * bn_grads[i]['beta']

    def train_epoch(self, X, y, lr=0.01, batch_size=256, progress_callback=None):
        """Train the network for one epoch with improved batch processing.
        
        Args:
            X: Input features
            y: Target labels
            lr: Learning rate
            batch_size: Size of training batches (increased for better stability)
            progress_callback: Optional callback function to report batch progress
        """
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        
        total_loss = 0
        num_batches = (X.shape[0] - 1) // batch_size + 1
        
        for batch_idx, start_idx in enumerate(range(0, X.shape[0], batch_size)):
            batch_indices = indices[start_idx:start_idx + batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # Forward pass with dropout
            activations, dropout_masks = self.forward(X_batch, training=True)
            
            # Backward pass
            self.backward(X_batch, y_batch, activations, dropout_masks, lr)
            
            # Calculate loss
            log_probs = np.log(activations[-1] + 1e-8)
            total_loss -= np.sum(y_batch * log_probs)
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(batch_idx + 1)
            
        return total_loss / X.shape[0]

    def predict(self, X):
        """Predict class labels for input X using test-time averaging."""
        activations = self.forward(X, training=False)
        return np.argmax(activations[-1], axis=1)

    def _batch_norm_backward(self, dout, x, bn_param):
        """Backward pass for batch normalization layer."""
        N = x.shape[0]
        
        # Get stored parameters
        gamma = bn_param['gamma']
        beta = bn_param['beta']
        xnorm = (x - bn_param['running_mean']) / np.sqrt(bn_param['running_var'] + 1e-5)
        
        # Gradients for gamma and beta
        dgamma = np.sum(dout * xnorm, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
        
        return {'gamma': dgamma, 'beta': dbeta}

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