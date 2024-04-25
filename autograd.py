import numpy as np

class Tensor(object):
    """
    Autograd-enabled tensor class for automatic differentiation.
    """
    def __init__(self, data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):
        """
        Initializes a tensor object.

        Args:
        - data (numpy.ndarray): The data contained in the tensor.
        - autograd (bool): Whether to enable autograd for this tensor.
        - creators (list of Tensor, optional): List of tensors that created this tensor.
        - creation_op (str, optional): The operation used to create this tensor.
        - id (int, optional): The unique identifier for this tensor.

        Returns:
        - None
        """
        self.data = np.array(data)
        self.autograd = autograd
        self.grad = None
        self.creation_op = creation_op
        self.creators = creators
        self.children = {}

        if id is None:
            self.id = np.random.randint(0, 1000000000)
        else:
            self.id = id

        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        """
        Checks if gradients from all children have been accounted for.

        Args:
        - None

        Returns:
        - bool: True if gradients from all children are accounted for, False otherwise.
        """
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True 

    def backward(self, grad=None, grad_origin=None):
        """
        Performs backpropagation to compute gradients.

        Args:
        - grad (Tensor, optional): Gradient of the current tensor.
        - grad_origin (Tensor, optional): Tensor that originated the gradient.

        Returns:
        - None
        """
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))

            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    return
                else:
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            assert grad.autograd == False

            if self.creators is not None and (
                self.all_children_grads_accounted_for() or grad_origin is None):
                
                if self.creation_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                    
                if self.creation_op == "sub":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(), self)

                if self.creation_op == "mul":
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new , self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)                    

                if self.creation_op == "mm":
                    c0 = self.creators[0]
                    c1 = self.creators[1]
                    new = self.grad.mm(c1.transpose())
                    c0.backward(new)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(new)

                if self.creation_op == "transpose":
                    self.creators[0].backward(self.grad.transpose())

                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim,
                                                               self.creators[0].data.shape[dim]))

                if "expand" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

                if self.creation_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())

                if self.creation_op == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))

                if self.creation_op == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))

                if self.creation_op == "relu":
                    self.creators[0].backward(self.grad * (self.data > 0))

                if self.creation_op == "index_select":
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))

                if self.creation_op == "cross_entropy":
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))


    def __add__(self, other):
        """
        Performs element-wise addition with another tensor.

        Args:
        - other (Tensor): The other tensor to add.

        Returns:
        - Tensor: The result of the addition.
        """
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="add")
        return Tensor(self.data + other.data)

    def __neg__(self):
        """
        Negates the tensor.

        Args:
        - None

        Returns:
        - Tensor: The negated tensor.
        """
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)
    
    def __sub__(self, other):
        """
        Performs element-wise subtraction with another tensor.

        Args:
        - other (Tensor): The other tensor to subtract.

        Returns:
        - Tensor: The result of the subtraction.
        """
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="sub")
        return Tensor(self.data - other.data)
    
    def __mul__(self, other):
        """
        Performs element-wise multiplication with another tensor.

        Args:
        - other (Tensor): The other tensor to multiply.

        Returns:
        - Tensor: The result of the multiplication.
        """
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul")
        return Tensor(self.data * other.data)    

    def sum(self, dim):
        """
        Computes the sum along the specified dimension.

        Args:
        - dim (int): The dimension along which to compute the sum.

        Returns:
        - Tensor: The result of the sum operation.
        """
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))
    
    def expand(self, dim, copies):
        """
        Expands the tensor along the specified dimension.

        Args:
        - dim (int): The dimension along which to expand.
        - copies (int): The number of copies to create.

        Returns:
        - Tensor: The expanded tensor.
        """
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)
        
        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_"+str(dim))
        return Tensor(new_data)
    
    def transpose(self):
        """
        Transposes the tensor.

        Args:
        - None

        Returns:
        - Tensor: The transposed tensor.
        """
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        
        return Tensor(self.data.transpose())
    
    def mm(self, x):
        """
        Performs matrix multiplication with another tensor.

        Args:
        - x (Tensor): The other tensor to multiply.

        Returns:
        - Tensor: The result of the matrix multiplication.
        """
        if self.autograd:
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))
    
    def reshape(self, *shape):
        """
        Reshapes the tensor to the specified shape.

        Args:
        - shape (tuple of int): The desired shape of the tensor.

        Returns:
        - Tensor: The reshaped tensor.
        """
        if self.autograd:
            return Tensor(self.data.reshape(*shape),
                        autograd=True,
                        creators=[self],
                        creation_op="view")
        return Tensor(self.data.reshape(*shape))
    
    @property
    def shape(self):
        """
        Returns the shape of the tensor.

        Returns:
        - tuple: The shape of the tensor.
        """
        return self.data.shape


    
    def sigmoid(self):
        """
        Applies the sigmoid function element-wise.

        Args:
        - None

        Returns:
        - Tensor: The result of the sigmoid operation.
        """
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        """
        Applies the hyperbolic tangent function element-wise.

        Args:
        - None

        Returns:
        - Tensor: The result of the tanh operation.
        """
        if self.autograd:
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))
    
    def relu(self):
        """
        Applies the Rectified Linear Unit (ReLU) activation function.

        Returns:
        - Tensor: Output tensor after applying the ReLU activation.
        """
        if self.autograd:
            return Tensor(np.maximum(0, self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="relu")
        return Tensor(np.maximum(0, self.data))


    
    def index_select(self, indices):
        """
        Selects elements from the tensor using the specified indices.

        Args:
        - indices (Tensor): The tensor containing the indices.

        Returns:
        - Tensor: The tensor containing the selected elements.
        """
        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])
    
    def softmax(self):
        """
        Computes the softmax function along the last dimension.

        Args:
        - None

        Returns:
        - Tensor: The result of the softmax operation.
        """
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        return softmax_output
    
    def cross_entropy(self, target_indices):
        """
        Computes the cross-entropy loss.

        Args:
        - target_indices (Tensor): The tensor containing target indices.

        Returns:
        - Tensor: The cross-entropy loss.
        """
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                    axis=len(self.data.shape)-1,
                                    keepdims=True)
        
        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * target_dist).sum(1).mean()

        if self.autograd:
            out = Tensor(loss,
                        autograd=True,
                        creators=[self],
                        creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out

        return Tensor(loss)

    def __repr__(self):
        """
        Returns a string representation of the tensor.

        Args:
        - None

        Returns:
        - str: The string representation of the tensor.
        """
        return str(self.data.__repr__())
    
    def __str__(self):
        """
        Returns a string representation of the tensor.

        Args:
        - None

        Returns:
        - str: The string representation of the tensor.
        """
        return str(self.data.__str__())  

