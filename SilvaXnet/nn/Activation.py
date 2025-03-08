import cupy as cp
from base import Activation  # Use the Activation base class

class Relu(Activation):
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
        return grad_output * (self.x > 0)  # Derivative of ReLU is 1 for x > 0, 0 otherwise


class Sigmoid(Activation):
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
        self.sigmoid_output = 1.0 / (1.0 + cp.exp(-x))
        return self.sigmoid_output

    def backward(self, grad_output):
        """
        Backward pass for the Sigmoid activation.
        
        Args:
            grad_output (cupy.ndarray): Gradient of the loss with respect to the output.
        
        Returns:
            cupy.ndarray: Gradient of the loss with respect to the input.
        """
        sigmoid_grad = self.sigmoid_output * (1 - self.sigmoid_output)
        return grad_output * sigmoid_grad  # Chain rule: grad_output * sigmoid'(x)


class Tanh(Activation):
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
        return grad_output * (1 - self.a ** 2)  # Derivative of Tanh: 1 - tanh(x)^2


class LeakyRelu(Activation):
    """
    Implements the Leaky ReLU activation function.
    
    Attributes:
        leaky_slope (float): The slope of the activation function for x <= 0.
    """
    def __init__(self, leaky_slope=0.01):
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
        return cp.where(x > 0, x, self.leaky_slope * x)  # Leaky ReLU: max(leaky_slope * x, x)

    def backward(self, grad_output):
        """
        Backward pass for the Leaky ReLU activation.
        
        Args:
            grad_output (cupy.ndarray): Gradient of the loss with respect to the output.
        
        Returns:
            cupy.ndarray: Gradient of the loss with respect to the input.
        """
        grad_input = cp.ones_like(self.x)
        grad_input[self.x <= 0] = self.leaky_slope
        return grad_output * grad_input  # Derivative of Leaky ReLU: 1 for x > 0, leaky_slope for x <= 0
