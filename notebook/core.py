import cupy as cp

class Layer:
    def __init__(self):
        pass 

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError
    

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, initializer: str = "he"):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights
        if initializer == 'he':
            scale = cp.sqrt(2.0 / in_features)
        elif initializer == 'xavier':
            scale = cp.sqrt(1.0 / in_features)
        else:
            scale = 1.0

        self.weights = cp.random.randn(out_features, in_features) * scale
        self.bias = cp.zeros((out_features,)) if bias else None

        # Initialize gradients
        self.dweights = cp.zeros_like(self.weights)
        self.dbias = cp.zeros_like(self.bias) if bias else None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """ Forward pass: Computes Y = XW^T + b """
        self.x = x  # Store input for backprop
        return cp.dot(x, self.weights.T) + (self.bias if self.bias is not None else 0)

    def backward(self, upstream_grad: cp.ndarray) -> cp.ndarray:
        """
        Backward pass (backpropagation)

        Args:
            upstream_grad: Gradient from subsequent layer, shape (batch_size, output_dim)

        Returns:
            Gradient with respect to input, shape (batch_size, input_dim)
        """
        # Compute gradients
        self.dweights = cp.dot(upstream_grad.T, self.x)  # (out_features, in_features)
        if self.bias is not None:
            self.dbias = cp.sum(upstream_grad, axis=0)    # (out_features,)

        # Compute gradient for input
        dx = cp.dot(upstream_grad, self.weights)         # (batch_size, in_features)

        return dx

    def update(self, learning_rate: float):
        """Update weights using computed gradients"""
        self.weights -= learning_rate * self.dweights
        if self.bias is not None:
            self.bias -= learning_rate * self.dbias

    @property
    def parameters(self):
        """Return weights and biases"""
        return {'weights': self.weights, 'bias': self.bias}

    @property
    def gradients(self):
        """Return current gradients"""
        return {'dweights': self.dweights, 'dbias': self.dbias}

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return cp.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + cp.exp(-x))

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(cp.float32)

    @staticmethod
    def sigmoid_derivative(x):
        sig = ActivationFunctions.sigmoid(x)
        return sig * (1 - sig)
    
    @staticmethod
    def tanh(x):
        return cp.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - cp.tanh(x)**2

   

class Activation:
    def __init__(self, activation: str):
        self.activation = activation
        self.x = None  # Store input for backprop

        # Set activation function and its derivative
        if activation == "relu":
            self.func = ActivationFunctions.relu
            self.derivative = ActivationFunctions.relu_derivative
        elif activation == "sigmoid":
            self.func = ActivationFunctions.sigmoid
            self.derivative = ActivationFunctions.sigmoid_derivative
        elif activation == "tanh":
            self.func = ActivationFunctions.tanh
            self.derivative = ActivationFunctions.tanh_derivative
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        """ Forward pass: Apply activation function """
        self.x = x  # Store for backward pass
        return self.func(x)

    def backward(self, upstream_grad):
        """ Backward pass: Apply activation derivative """
        return upstream_grad * self.derivative(self.x)


class Sequential:
    def __init__(self, *layers):
        """
        A simple sequential model to stack layers.

        Args:
            *layers: A list of layers (Linear, Activation, etc.)
        """
        self.layers = layers

    def forward(self, x):
        """ Forward pass through all layers """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, upstream_grad):
        """ Backward pass through all layers in reverse order """
        for layer in reversed(self.layers):
            upstream_grad = layer.backward(upstream_grad)
        return upstream_grad

    def update(self, learning_rate):
        """ Update weights of layers that have parameters (Linear layers) """
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(learning_rate)


class BCE:
    def __init__(self):
        pass
    
    def forward(self, output, target):
        # Binary Cross-Entropy loss: -target * log(output) - (1 - target) * log(1 - output)
        epsilon = 1e-15  # To avoid log(0) errors
        output = cp.clip(output, epsilon, 1 - epsilon)
        loss = -cp.mean(target * cp.log(output) + (1 - target) * cp.log(1 - output))
        self.output = output
        self.target = target
        return loss
    
    def backward(self):
        # Gradient of BCE loss with respect to output
        grad_input = (self.output - self.target) / (self.output * (1 - self.output))
        return grad_input




def train(model, loss_fn, x , y, epochs,learning_rate):
        # Training Loop
    for epoch in range(epochs):
        # Forward pass
        output = model.forward(x)

        # Compute loss
        loss = loss_fn.forward(output, y)

        # Backward pass
        upstream_grad = loss_fn.backward()  # Gradient of loss w.r.t. output
        model.backward(upstream_grad)

        # Update weights
        model.update(learning_rate)

        # Print loss for every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    return loss

