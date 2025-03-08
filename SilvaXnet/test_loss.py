# test_loss.py
import pytest
import cupy as cp
import numpy as np
from loss import MSE, CrossEntropy, Loss

# Helper functions
def cp_to_np(x):
    """Convert CuPy array to NumPy for assertions"""
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else x

def numerical_gradient(loss_func, y_true, y_pred, epsilon=1e-6):
    """
    Compute numerical gradient for a loss function.
    
    Args:
        loss_func: Loss function class
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small perturbation
    
    Returns:
        Numerical gradient
    """
    num_grad = cp.zeros_like(y_pred)
    
    for i in range(y_pred.size):
        # Flatten indices
        idx = np.unravel_index(i, y_pred.shape)
        
        # Save original value
        original = y_pred[idx].copy()
        
        # Compute loss with +epsilon
        y_pred[idx] = original + epsilon
        loss_plus = loss_func(y_true, y_pred)
        
        # Compute loss with -epsilon
        y_pred[idx] = original - epsilon
        loss_minus = loss_func(y_true, y_pred)
        
        # Restore original value
        y_pred[idx] = original
        
        # Compute numerical gradient
        num_grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
    
    return num_grad

class TestLoss:
    def test_mse_instantiation(self):
        """Test MSE instantiation with different parameters"""
        # Default parameters
        mse = MSE()
        assert mse.reduction == 'mean'
        assert mse.epsilon > 0
        
        # Custom parameters
        mse = MSE(reduction='sum', epsilon=1e-10)
        assert mse.reduction == 'sum'
        assert mse.epsilon == 1e-10
    
    def test_cross_entropy_instantiation(self):
        """Test CrossEntropy instantiation with different parameters"""
        # Default parameters
        ce = CrossEntropy()
        assert ce.reduction == 'mean'
        assert ce.epsilon > 0
        assert ce.weight is None
        assert ce.label_smoothing == 0.0
        
        # Custom parameters
        ce = CrossEntropy(reduction='sum', epsilon=1e-10, 
                          weight=cp.array([0.2, 0.8]), label_smoothing=0.1)
        assert ce.reduction == 'sum'
        assert ce.epsilon == 1e-10
        assert cp.allclose(ce.weight, cp.array([0.2, 0.8]))
        assert ce.label_smoothing == 0.1
    
    def test_mse_forward(self):
        """Test MSE forward pass with different inputs"""
        # Simple 1D case
        y_true = cp.array([1.0, 2.0, 3.0])
        y_pred = cp.array([1.5, 2.0, 2.5])
        
        # Test mean reduction
        mse_mean = MSE(reduction='mean')
        loss_mean = mse_mean(y_true, y_pred)
        expected_mean = ((1.5-1.0)**2 + (2.0-2.0)**2 + (2.5-3.0)**2) / 3
        assert np.isclose(cp_to_np(loss_mean), expected_mean)
        
        # Test sum reduction
        mse_sum = MSE(reduction='sum')
        loss_sum = mse_sum(y_true, y_pred)
        expected_sum = (1.5-1.0)**2 + (2.0-2.0)**2 + (2.5-3.0)**2
        assert np.isclose(cp_to_np(loss_sum), expected_sum)
        
        # Test no reduction
        mse_none = MSE(reduction='none')
        loss_none = mse_none(y_true, y_pred)
        expected_none = cp.array([(1.5-1.0)**2, (2.0-2.0)**2, (2.5-3.0)**2])
        assert cp.allclose(loss_none, expected_none)
        
        # Test with batch dimension
        y_true_batch = cp.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred_batch = cp.array([[1.5, 2.0], [3.0, 3.5]])
        
        mse_batch = MSE()
        loss_batch = mse_batch(y_true_batch, y_pred_batch)
        expected_batch = ((1.5-1.0)**2 + (2.0-2.0)**2 + (3.0-3.0)**2 + (3.5-4.0)**2) / 4
        assert np.isclose(cp_to_np(loss_batch), expected_batch)
        
        # Test for shape mismatch
        with pytest.raises(ValueError):
            mse = MSE()
            mse(cp.array([1.0, 2.0]), cp.array([1.0, 2.0, 3.0]))
    
    def test_binary_cross_entropy_forward(self):
        """Test binary CrossEntropy forward pass"""
        # Binary classification case
        y_true = cp.array([[1], [0], [1], [0]])
        y_pred = cp.array([[0.8], [0.3], [0.6], [0.1]])
        
        # Test mean reduction
        bce = CrossEntropy()
        loss = bce(y_true, y_pred)
        
        # Calculate expected loss
        expected = -(cp.log(0.8) + cp.log(0.7) + cp.log(0.6) + cp.log(0.9)) / 4
        assert np.isclose(cp_to_np(loss), cp_to_np(expected), rtol=1e-5)
        
        # Test with weights
        weights = cp.array([0.3, 0.7])  # Weight for class 0 and 1
        bce_weighted = CrossEntropy(weight=weights)
        loss_weighted = bce_weighted(y_true, y_pred)
        
        # Expected weighted loss
        expected_weighted = -((cp.log(0.8) * 0.7) + (cp.log(0.7) * 0.3) + 
                             (cp.log(0.6) * 0.7) + (cp.log(0.9) * 0.3)) / 4
        assert np.isclose(cp_to_np(loss_weighted), cp_to_np(expected_weighted), rtol=1e-5)
    
    def test_multiclass_cross_entropy_forward(self):
        """Test multiclass CrossEntropy forward pass"""
        # Multiclass classification case with one-hot encoding
        y_true = cp.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        y_pred = cp.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        # Test mean reduction
        ce = CrossEntropy()
        loss = ce(y_true, y_pred)
        
        # Calculate expected loss
        expected = -(cp.log(0.7) + cp.log(0.6) + cp.log(0.7)) / 3
        assert np.isclose(cp_to_np(loss), cp_to_np(expected), rtol=1e-5)
        
        # Test with class indices instead of one-hot
        y_true_indices = cp.array([0, 1, 2])
        ce_indices = CrossEntropy()
        loss_indices = ce_indices(y_true_indices, y_pred)
        
        # Should be same as with one-hot
        assert np.isclose(cp_to_np(loss_indices), cp_to_np(expected), rtol=1e-5)
        
        # Test with label smoothing
        ce_smooth = CrossEntropy(label_smoothing=0.1)
        loss_smooth = ce_smooth(y_true, y_pred)
        
        # Should be different from regular loss
        assert cp_to_np(loss) != cp_to_np(loss_smooth)
    
    def test_mse_backward(self):
        """Test MSE backward gradient computation"""
        # Simple case
        y_true = cp.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = cp.array([[1.5, 2.0], [3.0, 3.5]])
        
        # Compute analytical gradient
        mse = MSE()
        _ = mse(y_true, y_pred)  # Forward pass
        analytical_grad = mse.backward()
        
        # Compute numerical gradient
        num_grad = numerical_gradient(lambda t, p: MSE()(t, p), y_true, y_pred.copy())
        
        # Compare gradients
        assert cp.allclose(analytical_grad, num_grad, rtol=1e-4, atol=1e-4)
    
    def test_cross_entropy_backward(self):
        """Test CrossEntropy backward gradient computation"""
        # Binary case
        y_true_binary = cp.array([[1], [0]])
        y_pred_binary = cp.array([[0.8], [0.3]])
        
        # Compute analytical gradient for binary case
        bce = CrossEntropy()
        _ = bce(y_true_binary, y_pred_binary)
        analytical_grad_binary = bce.backward()
        
        # Compute numerical gradient
        num_grad_binary = numerical_gradient(
            lambda t, p: CrossEntropy()(t, p), 
            y_true_binary, 
            y_pred_binary.copy()
        )
        
        # Compare gradients (binary)
        assert cp.allclose(analytical_grad_binary, num_grad_binary, rtol=1e-4, atol=1e-4)
        
        # Multiclass case
        y_true_multi = cp.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        y_pred_multi = cp.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.2, 0.7]
        ])
        
        # Compute analytical gradient for multiclass
        ce = CrossEntropy()
        _ = ce(y_true_multi, y_pred_multi)
        analytical_grad_multi = ce.backward()
        
        # Compute numerical gradient
        num_grad_multi = numerical_gradient(
            lambda t, p: CrossEntropy()(t, p), 
            y_true_multi, 
            y_pred_multi.copy()
        )
        
        # Compare gradients (multiclass)
        assert cp.allclose(analytical_grad_multi, num_grad_multi, rtol=1e-4, atol=1e-4)
    
    def test_input_validation(self):
        """Test input validation and error handling"""
        # Empty arrays
        with pytest.raises(ValueError):
            mse = MSE()
            mse(cp.array([]), cp.array([]))
        
        with pytest.raises(ValueError):
            ce = CrossEntropy()
            ce(cp.array([]), cp.array([]))
        
        # Bad reduction type
        with pytest.raises(ValueError):
            mse = MSE(reduction='invalid')
            mse(cp.array([1.0]), cp.array([1.5]))
        
        # Call backward before forward
        with pytest.raises(RuntimeError):
            mse = MSE()
            mse.backward()
        
        with pytest.raises(RuntimeError):
            ce = CrossEntropy()
            ce.backward()
    
    def test_loss_as_base_class(self):
        """Test that our loss classes inherit from the abstract base class"""
        assert issubclass(MSE, Loss)
        assert issubclass(CrossEntropy, Loss)
        
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            loss = Loss()

# Run with: pytest test_loss.py -v