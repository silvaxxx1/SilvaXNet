import sys
import os

# Add the parent directory (project root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autograd import Tensor
import numpy as np


class SGD(object):
    """
    Stochastic Gradient Descent (SGD) optimizer with momentum.
    """
    def __init__(self, parameters, alpha=0.1, momentum=None):
        """
        Initializes the SGD optimizer with momentum.

        Args:
        - parameters (list of Tensor): List of parameters to optimize.
        - alpha (float, optional): Learning rate.
        - momentum (float, optional): Momentum factor. If None, momentum is not used.

        Returns:
        - None
        """
        self.parameters = parameters
        self.alpha = alpha
        self.momentum = momentum
        if momentum is not None:
            self.velocity = [Tensor(np.zeros_like(p.data)) for p in self.parameters]
    
    def zero(self):
        """
        Resets gradients of all parameters to zero.

        Args:
        - None

        Returns:
        - None
        """
        for p in self.parameters:
            p.grad.data *= 0
        
    def step(self, zero=True):
        """
        Updates parameters using SGD with momentum.

        Args:
        - zero (bool, optional): Whether to zero out gradients after update.

        Returns:
        - None
        """
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                print(f"Gradient of parameter {i} is None!")
                continue  # Skip this parameter if gradient is None

            if self.momentum is not None:
                # Update velocity if momentum is used
                self.velocity[i].data = self.momentum * self.velocity[i].data - self.alpha * p.grad.data
                # Update parameters using velocity
                p.data += self.velocity[i].data
            else:
                # Update parameters without momentum
                p.data -= self.alpha * p.grad.data
            
            if zero:
                p.grad.data *= 0




class Adam(object):
    """
    Adam optimizer.
    """
    def __init__(self, parameters, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initializes the Adam optimizer.

        Args:
        - parameters (list of Tensor): List of parameters to optimize.
        - alpha (float, optional): Learning rate.
        - beta1 (float, optional): Exponential decay rate for the first moment estimates.
        - beta2 (float, optional): Exponential decay rate for the second moment estimates.
        - epsilon (float, optional): Small constant for numerical stability.

        Returns:
        - None
        """
        self.parameters = parameters
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = [Tensor(np.zeros_like(p.data)) for p in parameters]
        self.v = [Tensor(np.zeros_like(p.data)) for p in parameters]
        self.t = 0
    
    def zero(self):
        """
        Resets gradients of all parameters to zero.

        Args:
        - None

        Returns:
        - None
        """
        for p in self.parameters:
            p.grad.data *= 0
        
    def step(self, zero=True):
        """
        Updates parameters using Adam.

        Args:
        - zero (bool, optional): Whether to zero out gradients after update.

        Returns:
        - None
        """
        self.t += 1
        for i, p in enumerate(self.parameters):
            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * p.grad.data
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (p.grad.data ** 2)
            m_hat = self.m[i].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[i].data / (1 - self.beta2 ** self.t)
            p.data -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            if zero:
                p.grad.data *= 0

class RMSprop(object):
    """
    RMSprop optimizer.
    """
    def __init__(self, parameters, alpha=0.001, rho=0.9, epsilon=1e-8):
        """
        Initializes the RMSprop optimizer.

        Args:
        - parameters (list of Tensor): List of parameters to optimize.
        - alpha (float, optional): Learning rate.
        - rho (float, optional): Decay rate.
        - epsilon (float, optional): Small constant for numerical stability.

        Returns:
        - None
        """
        self.parameters = parameters
        self.alpha = alpha
        self.rho = rho
        self.epsilon = epsilon
        
        self.avg_sq_grad = [Tensor(np.zeros_like(p.data)) for p in parameters]
    
    def zero(self):
        """
        Resets gradients of all parameters to zero.

        Args:
        - None

        Returns:
        - None
        """
        for p in self.parameters:
            p.grad.data *= 0
        
    def step(self, zero=True):
        """
        Updates parameters using RMSprop.

        Args:
        - zero (bool, optional): Whether to zero out gradients after update.

        Returns:
        - None
        """
        for i, p in enumerate(self.parameters):
            self.avg_sq_grad[i].data = self.rho * self.avg_sq_grad[i].data + (1 - self.rho) * (p.grad.data ** 2)
            p.data -= self.alpha * p.grad.data / (np.sqrt(self.avg_sq_grad[i].data) + self.epsilon)
            
            if zero:
                p.grad.data *= 0

class Adagrad(object):
    """
    Adagrad optimizer.
    """
    def __init__(self, parameters, alpha=0.01, epsilon=1e-8):
        """
        Initializes the Adagrad optimizer.

        Args:
        - parameters (list of Tensor): List of parameters to optimize.
        - alpha (float, optional): Learning rate.
        - epsilon (float, optional): Small constant for numerical stability.

        Returns:
        - None
        """
        self.parameters = parameters
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.sum_sq_grad = [Tensor(np.zeros_like(p.data)) for p in parameters]
    
    def zero(self):
        """
        Resets gradients of all parameters to zero.

        Args:
        - None

        Returns:
        - None
        """
        for p in self.parameters:
            p.grad.data *= 0
        
    def step(self, zero=True):
        """
        Updates parameters using Adagrad.

        Args:
        - zero (bool, optional): Whether to zero out gradients after update.

        Returns:
        - None
        """
        for i, p in enumerate(self.parameters):
            self.sum_sq_grad[i].data += p.grad.data ** 2
            p.data -= self.alpha * p.grad.data / (np.sqrt(self.sum_sq_grad[i].data) + self.epsilon)
            
            if zero:
                p.grad.data *= 0



