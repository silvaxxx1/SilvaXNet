import cupy as cp

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return cp.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(cp.float32)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + cp.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        sig = ActivationFunctions.sigmoid(x)
        return sig * (1 - sig)

class NeuralNetwork:
    def __init__(self, layers, activation_functions, learning_rate=0.01):
        """
        layers: List containing the number of neurons in each layer.
        activation_functions: List of activation functions for each layer except the input layer.
        """
        self.num_layers = len(layers)
        self.layers = layers
        self.learning_rate = learning_rate
        
        if len(activation_functions) != (self.num_layers - 1):
            raise ValueError(f"Expected {self.num_layers - 1} activation functions, got {len(activation_functions)}")
        
        self.activation = activation_functions

        # Initialize weights and biases on GPU
        self.weights = [cp.random.randn(layers[i], layers[i + 1]) * 0.01 for i in range(self.num_layers - 1)]
        self.biases = [cp.zeros((1, layers[i + 1])) for i in range(self.num_layers - 1)]

    def forward(self, X):
        """
        Forward propagation.
        """
        A = X
        self.cache = {"A0": A}  # Store activations for backpropagation

        for l in range(1, self.num_layers):
            Z = cp.dot(A, self.weights[l - 1]) + self.biases[l - 1]
            self.cache[f"Z{l}"] = Z  # Cache Z for backpropagation

            if self.activation[l - 1] == 'relu':
                A = ActivationFunctions.relu(Z)
            elif self.activation[l - 1] == 'sigmoid':
                A = ActivationFunctions.sigmoid(Z)
            else:
                raise ValueError(f"Unsupported activation function: {self.activation[l - 1]}")

            self.cache[f"A{l}"] = A  # Cache activation

        return A

    def backward(self, X, y):
        """
        Backpropagation to update weights.
        """
        m = X.shape[0]
        grads = {}

        # Compute output layer error (Mean Squared Error)
        dA = (self.cache[f"A{self.num_layers - 1}"] - y) / m

        for l in reversed(range(1, self.num_layers)):
            Z = self.cache[f"Z{l}"]
            A_prev = self.cache[f"A{l - 1}"]

            if self.activation[l - 1] == 'relu':
                dZ = dA * ActivationFunctions.relu_derivative(Z)
            elif self.activation[l - 1] == 'sigmoid':
                dZ = dA * ActivationFunctions.sigmoid_derivative(Z)
            else:
                raise ValueError(f"Unsupported activation function: {self.activation[l - 1]}")

            grads[f"dW{l}"] = cp.dot(A_prev.T, dZ)
            grads[f"db{l}"] = cp.sum(dZ, axis=0, keepdims=True)
            dA = cp.dot(dZ, self.weights[l - 1].T)

        return grads

    def update_parameters(self, grads):
        """
        Gradient Descent Update.
        """
        for l in range(1, self.num_layers):
            self.weights[l - 1] -= self.learning_rate * grads[f"dW{l}"]
            self.biases[l - 1] -= self.learning_rate * grads[f"db{l}"]

    def train(self, X, y, epochs=1000):
        """
        Training loop.
        """
        loss_history = []

        for epoch in range(epochs):
            output = self.forward(X)
            loss = cp.mean((output - y) ** 2)  # Mean Squared Error loss
            loss_history.append(loss)

            grads = self.backward(X, y)
            self.update_parameters(grads)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

        return loss_history

# Example usage
if __name__ == "__main__":
    # Define network with 3 layers: input (2), hidden (3), output (1)
    layers = [2, 3, 1]
    activation_functions = ['relu', 'sigmoid']

    nn = NeuralNetwork(layers, activation_functions, learning_rate=0.1)

    # Dummy dataset moved to GPU
    X = cp.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
    y = cp.array([[0], [1], [1], [0]])  # XOR outputs

    # Train the network
    loss_history = nn.train(X, y, epochs=1000)
