import cupy as cp

class Layer:
    def __init__(self):
        pass 

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True, initializer="he"):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier/He initialization
        if initializer == 'he':
            scale = cp.sqrt(2.0 / in_features)
        else:
            scale = cp.sqrt(1.0 / in_features)
            
        self.weights = cp.random.randn(in_features, out_features) * scale
        self.bias = cp.zeros((1, out_features)) if bias else None
        
        # Initialize gradients
        self.dweights = None
        self.dbias = None
        self.x = None

    def forward(self, x):
        self.x = x
        return cp.dot(x, self.weights) + (self.bias if self.bias is not None else 0)

    def backward(self, upstream_grad):
        # Compute gradients
        self.dweights = cp.dot(self.x.T, upstream_grad)
        if self.bias is not None:
            self.dbias = cp.sum(upstream_grad, axis=0, keepdims=True)
            
        # Compute gradient w.r.t input
        return cp.dot(upstream_grad, self.weights.T)
    
    def update(self, learning_rate: float):
        self.weights -= learning_rate * self.dweights
        if self.bias is not None:
            self.bias -= learning_rate * self.dbias
        # Reset gradients after update
        self.dweights.fill(0)
        if self.bias is not None:
            self.dbias.fill(0)

    @property
    def parameters(self):
        return {'weights': self.weights, 'bias': self.bias}

    @property
    def gradients(self):
        return {'dweights': self.dweights, 'dbias': self.dbias}

# ActivationFunctions class remains the same
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
        self.x = None

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
        self.x = x
        return self.func(x)

    def backward(self, upstream_grad):
        return upstream_grad * self.derivative(self.x)

class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, upstream_grad):
        for layer in reversed(self.layers):
            upstream_grad = layer.backward(upstream_grad)
        return upstream_grad

    def update(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(learning_rate)

class BCE:
    def __init__(self):
        self.output = None
        self.target = None
    
    def forward(self, output, target):
        epsilon = 1e-15
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        
        # Clip outputs for numerical stability
        self.output = cp.clip(output, epsilon, 1 - epsilon)
        self.target = target
        
        # Compute BCE loss without mean reduction
        loss = -(target * cp.log(self.output) + (1 - target) * cp.log(1 - self.output))
        return cp.mean(loss)
    
    def backward(self):
        # Simple BCE gradient without batch size scaling
        return -(self.target / self.output - (1 - self.target) / (1 - self.output))

def train(model, loss_fn, x, y, epochs, learning_rate):
    x = cp.asarray(x)
    y = cp.asarray(y)
    
    losses = []
    for epoch in range(epochs):
        # Forward pass
        output = model.forward(x)
        
        # Compute loss
        loss = loss_fn.forward(output, y)
        losses.append(float(loss))
        
        # Backward pass
        grad = loss_fn.backward()
        model.backward(grad)
        
        # Update parameters
        model.update(learning_rate)
        
        if epoch % 100 == 0:
            predictions = (output > 0.5).astype(int)
            accuracy = cp.mean((predictions == y).astype(float))
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return losses