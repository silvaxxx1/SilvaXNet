import cupy as cp
from base import Layer

class Linear(Layer):
    """
    A fully connected (dense) linear layer in a neural network, with optional bias and weight initialization.
    
    Parameters:
        in_features (int): The number of input features (size of input vector).
        out_features (int): The number of output features (size of output vector).
        bias (bool, optional): Whether to include a bias term (default is True).
        initializer (str, optional): Initialization method for weights. Options: 'he', 'xavier', or 'plain' (default is 'he').
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, initializer: str = "he"):
        """
        Initializes the Linear layer with weights, bias (if applicable), and chosen initialization method.
        
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool, optional): Whether to include a bias term. Default is True.
            initializer (str, optional): Initialization method for weights. Can be 'he', 'xavier', or 'plain'. Default is 'he'.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Select weight initialization based on the method chosen
        if initializer == 'he':
            scale = cp.sqrt(2.0 / in_features)
        elif initializer == 'xavier':
            scale = cp.sqrt(1.0 / in_features)
        elif initializer == 'plain':
            scale = 1.0
        else:
            raise ValueError("Invalid initializer. Choose from 'he', 'xavier', or 'plain'.")

        self.weights = cp.random.randn(in_features, out_features) * scale
        self.bias = cp.zeros((1, out_features)) if bias else None

        # Update params and gradients list to handle the case when bias is False
        self.params = [self.weights, self.bias] if self.bias is not None else [self.weights]
        self.grad = [cp.zeros_like(self.weights), cp.zeros_like(self.bias)] if self.bias is not None else [cp.zeros_like(self.weights)]

    def forward(self, x):
        """
        Performs the forward pass of the linear layer. It computes the output by applying 
        the linear transformation to the input and adds the bias if it exists.
        
        Args:
            x (cupy.ndarray): Input tensor of shape (batch_size, in_features).
        
        Returns:
            cupy.ndarray: Output tensor of shape (batch_size, out_features).
        """
        self.x = x  # Store input for backward pass
        print(f"Input shape: {x.shape}")  # Debugging line
        
        # Flatten the input except for the batch dimension
        prod_shape = cp.prod(cp.array(x.shape[1:])).item()  # Get the product of dimensions excluding batch size
        print(f"Prod of x.shape[1:]: {prod_shape}")  # Debugging line
        
        x_flatten = x.reshape(x.shape[0], prod_shape)  # Flatten input to (batch_size, flattened_size)
        
        # Compute the linear transformation (Wx + b)
        z = cp.dot(x_flatten, self.weights)
        if self.bias is not None:
            z = z + self.bias
        
        return z

    def backward(self, dZ):
        """
        Performs the backward pass of the linear layer. It computes the gradients of the 
        weights, bias, and input with respect to the loss during backpropagation.
        
        Args:
            dZ (cupy.ndarray): Gradient of the loss with respect to the output (shape: batch_size, out_features).
        
        Returns:
            cupy.ndarray: Gradient of the loss with respect to the input (shape: batch_size, in_features).
        """
        x = self.x  # Input from the forward pass
        original_shape = x.shape  # Store the original input shape for later reshaping
        prod_shape = cp.prod(cp.array(x.shape[1:])).item()  # Flatten input (exclude batch dimension)
        x_flatten = x.reshape(x.shape[0], prod_shape)  # Flatten input to (batch_size, flattened_size)

        # Compute the gradient with respect to weights
        dW = cp.dot(x_flatten.T, dZ)
        
        # Compute the gradient with respect to the bias, if bias exists
        if self.bias is not None:
            db = cp.sum(dZ, axis=0, keepdims=True)
            self.grad[1] += db
        
        # Compute the gradient with respect to the input
        dx_flat = cp.dot(dZ, self.weights.T)
        
        # Reshape the gradient with respect to the input back to the original shape
        dx = dx_flat.reshape(original_shape)

        # Store the gradients for later updates
        self.grad[0] += dW
        
        return dx  # Return the reshaped gradient for backpropagation

    def reg_grad(self, reg):
        """
        Computes the regularization gradient and adds it to the existing gradients.
        
        Args:
            reg (float): Regularization strength (lambda).
        """
        self.grad[0] += 2 * reg * self.weights  # L2 regularization on weights

    def reg_loss(self, reg):
        """
        Computes the regularization loss (L2 regularization) for this layer.
        
        Args:
            reg (float): Regularization strength (lambda).
        
        Returns:
            float: The regularization loss.
        """
        return reg * cp.sum(self.weights**2)  # L2 regularization loss
