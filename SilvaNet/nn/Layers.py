import numpy as np
from autograd import Tensor


class Layer(object):
    """
    Base class representing a neural network layer.

    Attributes:
    - parameters (list): List of parameters associated with the layer.
    """

    def __init__(self):
        """
        Initializes the layer.

        Args:
        - None

        Returns:
        - None
        """
        self.parameters = list()

    def get_parameters(self):
        """
        Retrieves the parameters associated with the layer.

        Args:
        - None

        Returns:
        - list: List of parameters.
        """
        return self.parameters

class Linear(Layer):
    """
    Linear layer implementation for neural networks.

    Args:
    - n_inputs (int): Number of input features.
    - n_outputs (int): Number of output features.
    - bias (bool, optional): Whether to include bias. Default is True.

    Attributes:
    - use_bias (bool): Indicates whether bias is used.
    - weight (Tensor): Learnable weights of the layer.
    - bias (Tensor or None): Learnable bias terms of the layer if use_bias is True, otherwise None.
    """

    def __init__(self, n_inputs, n_outputs, bias=True):
        """
        Initializes the linear layer.

        Args:
        - n_inputs (int): Number of input features.
        - n_outputs (int): Number of output features.
        - bias (bool, optional): Whether to include bias. Default is True.

        Returns:
        - None
        """
        super().__init__()
        
        self.use_bias = bias
        
        # Initialize weights with Xavier initialization
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0/(n_inputs))
        self.weight = Tensor(W, autograd=True)
        
        if self.use_bias:
            # Initialize bias terms if bias is enabled
            self.bias = Tensor(np.zeros(n_outputs), autograd=True)
        
        # Add parameters to the list
        self.parameters.append(self.weight)
        
        if self.use_bias:        
            self.parameters.append(self.bias)

    def forward(self, input):
        """
        Performs forward pass through the linear layer.

        Args:
        - input (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor after forward pass.
        """
        if self.use_bias:
            return input.mm(self.weight) + self.bias.expand(0, len(input.data))
        else:
            return input.mm(self.weight)

class Sequential(Layer):
    """
    Sequential layer implementation for neural networks.
    """
    def __init__(self, layers=list()):
        """
        Initializes the Sequential layer.

        Args:
        - layers (list): List of layers to be added to the Sequential model. Default is an empty list.

        Returns:
        - None
        """
        super().__init__()
        
        self.layers = layers
    
    def add(self, layer):
        """
        Adds a layer to the Sequential model.

        Args:
        - layer (Layer): Layer to be added to the Sequential model.

        Returns:
        - None
        """
        self.layers.append(layer)
        
    def forward(self, input):
        """
        Performs forward pass through the Sequential model.

        Args:
        - input (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor after forward pass through all layers.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def get_parameters(self):
        """
        Gets the parameters of the Sequential model.

        Returns:
        - list: List of parameters of all layers in the Sequential model.
        """
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params


class Embedding(Layer):
    """
    Embedding layer class.
    Inherits from the base layer class.
    """

    def __init__(self, vocab_size, dim):
        """
        Initializes the Embedding layer.

        Args:
        - vocab_size (int): The size of the vocabulary.
        - dim (int): The dimensionality of the embeddings.

        Returns:
        - None
        """
        super().__init__()
        
        # Set vocabulary size and embedding dimension
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Randomly initialize embedding weights using a convention from word2vec
        self.weight = Tensor((np.random.rand(vocab_size, dim) - 0.5) / dim, autograd=True)
        
        # Add the embedding weights to the list of parameters
        self.parameters.append(self.weight)
    
    def forward(self, input):
        """
        Performs the forward pass through the Embedding layer.

        Args:
        - input (Tensor): The input tensor representing indices of words.

        Returns:
        - Tensor: The output tensor containing the embeddings for the input words.
        """
        return self.weight.index_select(input)


from nn.activations import Sigmoid , Tanh    

class RNNCell(Layer):
    """
    RNN Cell layer class.
    Inherits from the base layer class.
    """

    def __init__(self, n_inputs, n_hidden, n_output, activation='sigmoid'):
        """
        Initializes the RNN Cell layer.

        Args:
        - n_inputs (int): Number of input features.
        - n_hidden (int): Number of hidden units.
        - n_output (int): Number of output features.
        - activation (str, optional): Activation function to use ('sigmoid' or 'tanh').

        Returns:
        - None
        """
        super().__init__()

        # Set the sizes of inputs, hidden units, and outputs
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        # Initialize the activation function
        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            raise Exception("Non-linearity not found")

        # Initialize the linear transformation matrices
        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)
        
        # Add the parameters of linear layers to the list of parameters
        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()        
    
    def forward(self, input, hidden):
        """
        Performs the forward pass through the RNN Cell layer.

        Args:
        - input (Tensor): The input tensor.
        - hidden (Tensor): The hidden state tensor from the previous time step.

        Returns:
        - output (Tensor): The output tensor.
        - new_hidden (Tensor): The new hidden state tensor.
        """
        # Calculate the contribution from the previous hidden state
        from_prev_hidden = self.w_hh.forward(hidden)
        # Combine the input and previous hidden state
        combined = self.w_ih.forward(input) + from_prev_hidden
        # Apply the activation function to the combined input
        new_hidden = self.activation.forward(combined)
        # Calculate the output
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden
    
    def init_hidden(self, batch_size=1):
        """
        Initializes the hidden state tensor with zeros.

        Args:
        - batch_size (int, optional): The batch size.

        Returns:
        - Tensor: The initialized hidden state tensor.
        """
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)


from nn.activations import Sigmoid, Tanh, ReLU

class Dense(Layer):
    """
    Dense layer implementation for neural networks.
    Inherits from the base layer class.
    """

    def __init__(self, n_inputs, n_units, activation='sigmoid'):
        """
        Initializes the Dense layer.

        Args:
        - n_inputs (int): Number of input features.
        - n_units (int): Number of units/neurons in the layer.
        - activation (str, optional): Activation function to use ('sigmoid', 'tanh', or 'relu'). Default is 'sigmoid'.

        Returns:
        - None
        """
        super().__init__()

        # Set the number of input features and units
        self.n_inputs = n_inputs
        self.n_units = n_units
        
        # Initialize the activation function
        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation = Tanh()
        elif activation == 'relu':
            self.activation = ReLU()
        else:
            raise Exception("Activation function not supported")

        # Initialize the linear transformation matrix
        self.linear = Linear(n_inputs, n_units)
        
        # Add the parameters of the linear layer to the list of parameters
        self.parameters += self.linear.get_parameters()

    def forward(self, inputs):
        """
        Perform forward pass through the Dense layer.

        Args:
        - inputs (Tensor): Input tensor to the layer.

        Returns:
        - outputs (Tensor): Output tensor from the layer.
        """
        # Perform linear transformation
        linear_output = self.linear.forward(inputs)

        # Apply activation function
        outputs = self.activation.forward(linear_output)

        return outputs

    def output_shape(self):
        """
        Get the output shape of the Dense layer.

        Returns:
        - output_shape (tuple): Shape of the output tensor.
        """
        return (self.n_units,)

    
    def forward(self, input):
        """
        Performs the forward pass through the Dense layer.

        Args:
        - input (Tensor): The input tensor.

        Returns:
        - Tensor: The output tensor after forward pass through the layer.
        """
        # Compute the linear transformation
        linear_output = self.linear.forward(input)
        # Apply the activation function
        output = self.activation.forward(linear_output)
        return output


class Dropout(Layer):
    """
    Dropout layer implementation for neural networks.
    Inherits from the base layer class.
    """

    def __init__(self, drop_prob):
        """
        Initializes the Dropout layer.

        Args:
        - drop_prob (float): The dropout probability (between 0 and 1).

        Returns:
        - None
        """
        super().__init__()

        # Set the dropout probability
        self.drop_prob = drop_prob
        self.training = True  # Initialize the training attribute to True by default

    def forward(self, input):
        """
        Performs forward pass through the Dropout layer.

        Args:
        - input (Tensor or np.ndarray): Input tensor or NumPy array.

        Returns:
        - Tensor: Output tensor after applying dropout.
        """
        if isinstance(input, Tensor):
            # If input is a Tensor, convert it to a NumPy array
            input = input.data
        
        if self.training:
            # Generate a mask with values sampled from a Bernoulli distribution
            mask_shape = np.shape(input)
            self.mask = np.random.binomial(1, 1 - self.drop_prob, size=mask_shape) / (1 - self.drop_prob)
            # Apply the mask to the input
            output = input * self.mask
        else:
            # During evaluation, scale the input by (1 - drop_prob) to keep the expected value the same
            output = input * (1 - self.drop_prob)
        return output

class Sequential(Layer):
    """
    Sequential layer implementation for neural networks.
    """

    def __init__(self, layers=list()):
        """
        Initializes the Sequential layer.

        Args:
        - layers (list): List of layers to be added to the Sequential model. Default is an empty list.

        Returns:
        - None
        """
        super().__init__()
        
        self.layers = layers
    
    def add(self, layer):
        """
        Adds a layer to the Sequential model.

        Args:
        - layer (Layer): Layer to be added to the Sequential model.

        Returns:
        - None
        """
        self.layers.append(layer)
        
    def forward(self, input):
        """
        Performs forward pass through the Sequential model.

        Args:
        - input (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor after forward pass through all layers.
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def get_parameters(self):
        """
        Gets the parameters of the Sequential model.

        Returns:
        - list: List of parameters of all layers in the Sequential model.
        """
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params


class Flatten(Layer):
    """
    Flatten layer implementation for neural networks.
    Inherits from the base layer class.
    """

    def __init__(self):
        """
        Initializes the Flatten layer.

        Args:
        - None

        Returns:
        - None
        """
        super().__init__()

    def forward(self, input):
        """
        Performs forward pass through the Flatten layer.

        Args:
        - input (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor after flattening.
        """
        return input.reshape((input.shape[0], -1))

