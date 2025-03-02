from autograd import Tensor

class CrossEntropyLoss(object):
    """
    Cross-entropy loss function.
    """
    def __init__(self):
        """
        Initializes the CrossEntropyLoss.

        Args:
        - None

        Returns:
        - None
        """
        super().__init__()
    
    def forward(self, input, target):
        """
        Computes the forward pass of the cross-entropy loss.

        Args:
        - input (Tensor): Predicted output tensor.
        - target (Tensor): Target tensor.

        Returns:
        - Tensor: Cross-entropy loss value.
        """
        return input.cross_entropy(target)

class MSELoss(object):
    """
    Mean squared error (MSE) loss function.
    """
    def __init__(self):
        """
        Initializes the MSELoss.

        Args:
        - None

        Returns:
        - None
        """
        super().__init__()
    
    def forward(self, input, target):
        """
        Computes the forward pass of the mean squared error (MSE) loss.

        Args:
        - input (Tensor): Predicted output tensor.
        - target (Tensor): Target tensor.

        Returns:
        - Tensor: Mean squared error (MSE) loss value.
        """
        dif = input - target
        return (dif * dif).sum(0)
