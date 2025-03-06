import cupy as cp
from base import Layer 


class Relu(Layer):
    """
    Implements the ReLU activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass for the ReLU activation.
        
        Args:
            x (cupy.ndarray): Input tensor.
        
        Returns:
            cupy.ndarray: The result after applying ReLU (max(0, x)).
        """
        self.x = x
        return cp.maximum(0, x)

    def backward(self, grad_output):
        """
        Backward pass for the ReLU activation.
        
        Args:
            grad_output (cupy.ndarray): Gradient of the loss with respect to the output.
        
        Returns:
            cupy.ndarray: Gradient of the loss with respect to the input.
        """
        x = self.x
        relu_grad = x > 0
        return grad_output * relu_grad 


class Sigmoid(Layer):
    """
    Implements the Sigmoid activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass for the Sigmoid activation.
        
        Args:
            x (cupy.ndarray): Input tensor.
        
        Returns:
            cupy.ndarray: The result after applying Sigmoid (1 / (1 + exp(-x))).
        """
        self.x = x
        return 1.0 / (1.0 + cp.exp(-x))

    def backward(self, grad_output):
        """
        Backward pass for the Sigmoid activation.
        
        Args:
            grad_output (cupy.ndarray): Gradient of the loss with respect to the output.
        
        Returns:
            cupy.ndarray: Gradient of the loss with respect to the input.
        """
        x = self.x
        a = 1.0 / (1.0 + cp.exp(-x))
        return grad_output * a * (1 - a)


class Tanh(Layer):
    """
    Implements the Tanh activation function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass for the Tanh activation.
        
        Args:
            x (cupy.ndarray): Input tensor.
        
        Returns:
            cupy.ndarray: The result after applying Tanh (tanh(x)).
        """
        self.x = x
        self.a = cp.tanh(x)
        return self.a

    def backward(self, grad_output):
        """
        Backward pass for the Tanh activation.
        
        Args:
            grad_output (cupy.ndarray): Gradient of the loss with respect to the output.
        
        Returns:
            cupy.ndarray: Gradient of the loss with respect to the input.
        """
        d = 1 - cp.square(self.a)
        return grad_output * d


class Leaky_relu(Layer):
    """
    Implements the Leaky ReLU activation function.
    
    Attributes:
        leaky_slope (float): The slope of the activation function for x <= 0.
    """
    def __init__(self, leaky_slope):
        """
        Initializes the Leaky ReLU activation.
        
        Args:
            leaky_slope (float): The slope of the activation function for x <= 0.
        """
        super().__init__()
        self.leaky_slope = leaky_slope

    def forward(self, x):
        """
        Forward pass for the Leaky ReLU activation.
        
        Args:
            x (cupy.ndarray): Input tensor.
        
        Returns:
            cupy.ndarray: The result after applying Leaky ReLU (max(leaky_slope * x, x)).
        """
        self.x = x
        return cp.maximum(self.leaky_slope * x, x)

    def backward(self, grad_output):
        """
        Backward pass for the Leaky ReLU activation.
        
        Args:
            grad_output (cupy.ndarray): Gradient of the loss with respect to the output.
        
        Returns:
            cupy.ndarray: Gradient of the loss with respect to the input.
        """
        x = self.x
        d = cp.zeros_like(x)
        d[x <= 0] = self.leaky_slope
        d[x > 0] = 1
        return grad_output * d
