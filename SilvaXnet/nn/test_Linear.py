# test_linear.py

import pytest
import cupy as cp
import numpy as np
from base import Layer  # Assuming this exists
from Linear import Linear  # Assuming your Linear class is in linear.py

# Create a helper function to convert CuPy arrays to NumPy for assertions
def cp_to_np(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else x

class TestLinear:
    def test_initialization(self):
        """Test initialization with different parameters"""
        # Test with bias
        layer = Linear(10, 5, bias=True)
        assert layer.weights.shape == (10, 5)
        assert layer.bias.shape == (1, 5)
        assert len(layer.params) == 2
        assert len(layer.grad) == 2
        
        # Test without bias
        layer_no_bias = Linear(10, 5, bias=False)
        assert layer_no_bias.weights.shape == (10, 5)
        assert layer_no_bias.bias is None
        assert len(layer_no_bias.params) == 1
        assert len(layer_no_bias.grad) == 1
        
        # Test different initializers
        layer_he = Linear(10, 5, initializer='he')
        layer_xavier = Linear(10, 5, initializer='xavier')
        layer_plain = Linear(10, 5, initializer='plain')
        
        # Different scalings should result in different weight magnitudes
        assert cp.mean(cp.abs(layer_he.weights)) != cp.mean(cp.abs(layer_xavier.weights))
        assert cp.mean(cp.abs(layer_xavier.weights)) != cp.mean(cp.abs(layer_plain.weights))
    
    def test_forward(self):
        """Test forward pass with different input shapes"""
        # Test with 2D input
        layer = Linear(10, 5)
        x = cp.random.randn(32, 10)  # batch size 32, 10 features
        output = layer.forward(x)
        assert output.shape == (32, 5)
        
        # Test with bias=False
        layer_no_bias = Linear(10, 5, bias=False)
        output_no_bias = layer_no_bias.forward(x)
        assert output_no_bias.shape == (32, 5)
        
        # Test with higher-dimensional input
        layer = Linear(12, 5)
        x_3d = cp.random.randn(32, 3, 4)  # 3×4=12 features when flattened
        output_3d = layer.forward(x_3d)
        assert output_3d.shape == (32, 5)
        
        # Test with even higher-dimensional input
        layer = Linear(16, 5)
        x_4d = cp.random.randn(32, 2, 2, 4)  # 2×2×4=16 features when flattened
        output_4d = layer.forward(x_4d)
        assert output_4d.shape == (32, 5)
    
    def test_backward(self):
        """Test backward pass and gradient computation"""
        # Test with 2D input
        layer = Linear(10, 5)
        x = cp.random.randn(32, 10)
        output = layer.forward(x)
        
        # Create a gradient for the output
        dZ = cp.random.randn(*output.shape)
        
        # Reset gradients
        layer.grad = [cp.zeros_like(layer.weights), cp.zeros_like(layer.bias)]
        
        # Run backward pass
        dx = layer.backward(dZ)
        
        # Check gradient shapes
        assert dx.shape == x.shape
        assert layer.grad[0].shape == layer.weights.shape
        assert layer.grad[1].shape == layer.bias.shape
        
        # Test with higher-dimensional input
        layer = Linear(12, 5)
        x_3d = cp.random.randn(32, 3, 4)
        output_3d = layer.forward(x_3d)
        dZ_3d = cp.random.randn(*output_3d.shape)
        
        # Reset gradients
        layer.grad = [cp.zeros_like(layer.weights), cp.zeros_like(layer.bias)]
        
        # Run backward pass
        dx_3d = layer.backward(dZ_3d)
        
        # Check that gradient has same shape as input
        assert dx_3d.shape == x_3d.shape
    
    def test_no_bias(self):
        """Test specifically the handling of no bias case"""
        layer = Linear(10, 5, bias=False)
        x = cp.random.randn(32, 10)
        output = layer.forward(x)
        
        dZ = cp.random.randn(*output.shape)
        
        # Reset gradients
        layer.grad = [cp.zeros_like(layer.weights)]
        
        # Run backward pass
        dx = layer.backward(dZ)
        
        # Check that only weight gradient exists
        assert len(layer.grad) == 1
        assert layer.grad[0].shape == layer.weights.shape
    
    def test_regularization(self):
        """Test regularization loss and gradient"""
        layer = Linear(10, 5)
        
        # Reset gradients
        layer.grad = [cp.zeros_like(layer.weights), cp.zeros_like(layer.bias)]
        
        # Test regularization loss
        reg = 0.01
        reg_loss = layer.reg_loss(reg)
        assert isinstance(reg_loss, (float, np.float64)) or isinstance(reg_loss, cp.ndarray)
        
        # Test regularization gradient
        layer.reg_grad(reg)
        # The gradient should not be zero after applying regularization
        assert cp.sum(cp.abs(layer.grad[0])) > 0
    
    def test_numerical_gradient(self):
        """Test gradient computation using numerical approximation"""
        layer = Linear(5, 3, bias=True)
        x = cp.random.randn(8, 5)
        
        # Forward pass
        output = layer.forward(x)
        
        # Create a dummy loss (MSE)
        y = cp.random.randn(*output.shape)
        loss = cp.mean((output - y) ** 2)
        
        # Compute gradient of loss w.r.t output
        dZ = 2 * (output - y) / output.size
        
        # Reset gradients
        layer.grad = [cp.zeros_like(layer.weights), cp.zeros_like(layer.bias)]
        
        # Backward pass
        layer.backward(dZ)
        
        # Numerical gradient checking for weights
        epsilon = 1e-6
        numerical_grad = cp.zeros_like(layer.weights)
        
        # Check a few random elements to save time
        indices = [(i, j) for i in range(0, 5, 2) for j in range(0, 3, 2)]
        
        for i, j in indices:
            # Add epsilon to the weight
            layer.weights[i, j] += epsilon
            output_plus = layer.forward(x)
            loss_plus = cp.mean((output_plus - y) ** 2)
            
            # Subtract epsilon from the weight
            layer.weights[i, j] -= 2 * epsilon
            output_minus = layer.forward(x)
            loss_minus = cp.mean((output_minus - y) ** 2)
            
            # Restore the weight
            layer.weights[i, j] += epsilon
            
            # Compute numerical gradient
            numerical_grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # Compare with analytical gradient for the checked indices
        for i, j in indices:
            assert np.isclose(cp_to_np(numerical_grad[i, j]), 
                              cp_to_np(layer.grad[0][i, j]), 
                              rtol=1e-3, atol=1e-3)

# Run with: pytest test_linear.py -v