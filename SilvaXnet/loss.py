import cupy as cp
import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    """Abstract base class for all loss functions."""
    
    @abstractmethod
    def __call__(self, y_true, y_pred):
        """Compute the loss value."""
        pass
    
    @abstractmethod
    def backward(self):
        """Compute the gradient of the loss with respect to y_pred."""
        pass
    
    def __repr__(self):
        """Return a string representation of the loss function."""
        return f"{self.__class__.__name__}()"


class MSE(Loss):
    """Mean Squared Error loss function."""
    
    def __init__(self, reduction='mean', epsilon=1e-12):
        """
        Initialize MSE loss.
        
        Args:
            reduction (str): How to reduce the loss - 'mean', 'sum', or 'none'
            epsilon (float): Small constant for numerical stability
        """
        self.reduction = reduction
        self.epsilon = epsilon
        self.y_true = None
        self.y_pred = None
        self.loss = None
        self.grad = None
        
    def __call__(self, y_true, y_pred):
        """
        Compute MSE loss between predictions and targets.
        
        Args:
            y_true (cp.ndarray): Ground truth values
            y_pred (cp.ndarray): Predicted values
            
        Returns:
            float or cp.ndarray: Loss value
        """
        # Ensure the shapes are compatible
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
        
        # Check for empty arrays
        if y_true.size == 0 or y_pred.size == 0:
            raise ValueError("Input arrays must not be empty")
        
        self.y_true = y_true
        self.y_pred = y_pred
        
        # Compute squared differences
        squared_diff = (y_pred - y_true) ** 2
        
        # Apply reduction
        if self.reduction == 'mean':
            self.loss = cp.mean(squared_diff)
        elif self.reduction == 'sum':
            self.loss = cp.sum(squared_diff)
        elif self.reduction == 'none':
            self.loss = squared_diff
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
            
        return self.loss
    
    def backward(self):
        """
        Compute gradient of MSE loss with respect to predictions.
        
        Returns:
            cp.ndarray: Gradient of loss w.r.t. predictions
        """
        if self.y_true is None or self.y_pred is None:
            raise RuntimeError("Cannot compute gradient before forward pass")
            
        # Basic gradient of squared error
        grad = 2 * (self.y_pred - self.y_true)
        
        # Apply reduction factor
        if self.reduction == 'mean':
            grad = grad / self.y_true.size
        
        self.grad = grad
        return grad


class CrossEntropy(Loss):
    """Cross Entropy loss for classification tasks."""
    
    def __init__(self, reduction='mean', epsilon=1e-12, weight=None, label_smoothing=0.0):
        """
        Initialize Cross Entropy loss.
        
        Args:
            reduction (str): How to reduce the loss - 'mean', 'sum', or 'none'
            epsilon (float): Small constant for numerical stability
            weight (cp.ndarray, optional): Weight for each class, for imbalanced datasets
            label_smoothing (float): Label smoothing value between 0 and 1
        """
        self.reduction = reduction
        self.epsilon = epsilon
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.y_true = None
        self.y_pred = None
        self.is_binary = None
        self.loss = None
        self.grad = None
        
    def __call__(self, y_true, y_pred):
        """
        Compute cross entropy loss between predictions and targets.
        
        Args:
            y_true (cp.ndarray): Ground truth values (one-hot or class indices)
            y_pred (cp.ndarray): Predicted probabilities
            
        Returns:
            float or cp.ndarray: Loss value
        """
        # Check for empty arrays
        if y_true.size == 0 or y_pred.size == 0:
            raise ValueError("Input arrays must not be empty")
            
        # Determine if this is binary classification
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
            self.is_binary = True
        else:
            self.is_binary = False
            
        # If y_true contains class indices rather than one-hot vectors
        if len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1 and not self.is_binary):
            # Convert to one-hot if needed
            if self.is_binary:
                # For binary classification, keep as is
                pass
            else:
                # For multi-class, convert to one-hot
                n_classes = y_pred.shape[1]
                
                if len(y_true.shape) == 2:
                    y_true = y_true.squeeze(1)  # Remove the extra dimension
                    
                # Create one-hot encoding
                one_hot = cp.zeros((y_true.shape[0], n_classes), dtype=y_pred.dtype)
                one_hot[cp.arange(y_true.shape[0]), y_true.astype(int)] = 1
                y_true = one_hot
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0 and not self.is_binary:
            n_classes = y_true.shape[1]
            soft_targets = (1.0 - self.label_smoothing) * y_true + self.label_smoothing / n_classes
            y_true = soft_targets
                
        self.y_true = y_true
        self.y_pred = y_pred
        
        # Clip predictions for numerical stability
        y_pred_safe = cp.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        
        # Compute loss
        if self.is_binary:
            # Binary cross entropy
            loss_per_sample = -(y_true * cp.log(y_pred_safe) + (1 - y_true) * cp.log(1 - y_pred_safe))
            
            # Apply class weights if provided
            if self.weight is not None:
                weight_per_sample = y_true * self.weight[1] + (1 - y_true) * self.weight[0]
                loss_per_sample = loss_per_sample * weight_per_sample
        else:
            # Multi-class cross entropy
            loss_per_sample = -cp.sum(y_true * cp.log(y_pred_safe), axis=1)
            
            # Apply class weights if provided
            if self.weight is not None:
                class_weights = cp.sum(y_true * self.weight, axis=1)
                loss_per_sample = loss_per_sample * class_weights
        
        # Apply reduction
        if self.reduction == 'mean':
            self.loss = cp.mean(loss_per_sample)
        elif self.reduction == 'sum':
            self.loss = cp.sum(loss_per_sample)
        elif self.reduction == 'none':
            self.loss = loss_per_sample
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
            
        return self.loss
    
    def backward(self):
        """
        Compute gradient of cross entropy loss with respect to predictions.
        
        Returns:
            cp.ndarray: Gradient of loss w.r.t. predictions
        """
        if self.y_true is None or self.y_pred is None:
            raise RuntimeError("Cannot compute gradient before forward pass")
            
        # Clip predictions for numerical stability
        y_pred_safe = cp.clip(self.y_pred, self.epsilon, 1.0 - self.epsilon)
        
        # Compute gradient
        if self.is_binary:
            # Binary classification gradient
            grad = -(self.y_true / y_pred_safe - (1 - self.y_true) / (1 - y_pred_safe))
            
            # Apply class weights if provided
            if self.weight is not None:
                weight_per_sample = self.y_true * self.weight[1] + (1 - self.y_true) * self.weight[0]
                grad = grad * weight_per_sample.reshape(-1, 1)
        else:
            # Multi-class classification gradient
            grad = -self.y_true / y_pred_safe
            
            # Apply class weights if provided
            if self.weight is not None:
                grad = grad * self.weight
        
        # Apply reduction factor
        if self.reduction == 'mean':
            grad = grad / self.y_true.shape[0]
        
        self.grad = grad
        return grad