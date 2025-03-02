from autograd import Tensor
from .Layers import Layer


class Tanh(Layer):
    """
    Hyperbolic tangent (tanh) activation function.
    """
    def __init__(self):
        """
        Initializes the Tanh activation function.

        Args:
        - None

        Returns:
        - None
        """
        super().__init__()
    
    def forward(self, input):
        """
        Performs forward pass through the Tanh activation function.

        Args:
        - input (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor after applying the tanh activation.
        """
        return input.tanh()


class Sigmoid(Layer):
    """
    Sigmoid activation function.
    """
    def __init__(self):
        """
        Initializes the Sigmoid activation function.

        Args:
        - None

        Returns:
        - None
        """
        super().__init__()
    
    def forward(self, input):
        """
        Performs forward pass through the Sigmoid activation function.

        Args:
        - input (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor after applying the sigmoid activation.
        """
        return input.sigmoid()


class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) activation function.
    """
    def __init__(self):
        """
        Initializes the ReLU activation function.

        Args:
        - None

        Returns:
        - None
        """
        super().__init__()
    
    def forward(self, input):
        """
        Performs forward pass through the ReLU activation function.

        Args:
        - input (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor after applying the ReLU activation.
        """
        return input.relu()


class Softmax(Layer):
    """
    Softmax activation function.
    """
    def __init__(self):
        """
        Initializes the Softmax activation function.

        Args:
        - None

        Returns:
        - None
        """
        super().__init__()
    
    def forward(self, input):
        """
        Performs forward pass through the Softmax activation function.

        Args:
        - input (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor after applying the softmax activation.
        """
        return input.softmax()
