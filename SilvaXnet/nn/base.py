from abc import ABC, abstractmethod
import cupy as cp

# Base class for all layers (e.g., Conv, Dense, Activation)
class Layer(ABC):
    def __init__(self):
        self.params = None  # Parameters for layers (e.g., weights, biases)
        self.cache = None  # Cache for the forward pass (needed for backprop)

    @abstractmethod
    def forward(self, x):
        """
        Defines the forward pass for the layer.
        """
        pass

    @abstractmethod
    def backward(self, x, grad):
        """
        Defines the backward pass for the layer (gradient calculation).
        """
        pass

    def reg_grad(self, reg):
        """
        Regularization gradient (could be L2 regularization for example).
        """
        pass

    def reg_loss(self, reg):
        """
        Returns the regularization loss (could be L2 regularization loss).
        """
        return 0


# Base class for models (Neural Network Model)
class Model(ABC):
    def __init__(self):
        self.layers = []  # List to store layers of the model
        self.metrics = []  # Metrics (accuracy, etc.)
        self.optimizer = None  # Optimizer (SGD, Adam, etc.)

    @abstractmethod
    def forward(self, input):
        """
        Defines the forward pass through the model.
        """
        pass

    @abstractmethod
    def backward(self, output):
        """
        Defines the backward pass through the model.
        """
        pass

    @abstractmethod
    def update(self, learning_rate):
        """
        Updates the parameters using the optimizer.
        """
        pass

    def compile(self, optimizer, metrics):
        """
        Compiles the model with the optimizer and metrics.
        """
        self.optimizer = optimizer
        self.metrics = metrics


# Base class for optimizers (e.g., SGD, Adam)
class Optimizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self):
        """
        Updates the parameters based on the gradients.
        """
        pass


# Base class for loss functions (e.g., CrossEntropy, MSE)
class Loss(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, output, target):
        """
        Computes the loss given the model output and the target.
        """
        pass

    @abstractmethod
    def backward(self):
        """
        Computes the gradient of the loss with respect to the model parameters.
        """
        pass


# Base class for activation functions (e.g., ReLU, Sigmoid)
from abc import ABC, abstractmethod
from base import Layer

# Base class for activation functions (e.g., ReLU, Sigmoid)
class Activation(Layer, ABC):
    """
    This class is now treated as a Layer to allow for consistency with other layers
    like Dense or Conv2D.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, input):
        """
        Forward pass for the activation function.
        """
        pass

    @abstractmethod
    def backward(self, output):
        """
        Backward pass for the activation function.
        """
        pass



# Base class for metrics (e.g., Accuracy, Precision)
class Metric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update(self, output, target):
        """
        Updates the metric with the current output and target.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the metric (e.g., at the start of a new epoch).
        """
        pass

    @abstractmethod
    def result(self):
        """
        Returns the current value of the metric.
        """
        pass
