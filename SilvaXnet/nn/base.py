# the base class for all layers 
# the base model class for all models 
# the base class for all optimizers
# the base class for all losses
# the base class for all activations

class Layer:

    def __init__(self):

        self.params = None

        pass

    def forward(self, x):       

        raise NotImplementedError

    def backward(self, x, grad):        

        raise NotImplementedError

    def reg_grad(self,reg):

        pass

    def reg_loss(self,reg):

        return 0.  


class Model:
    def __init__(self):
        pass

    def forward(self, input):
        pass

    def backward(self, output):
        pass

    def update(self, learning_rate):

        pass


class Optimizer:
    def __init__(self):
        pass

    def step(self):
        pass


class Loss:
    def __init__(self):
        pass

    def forward(self, output, target):
        pass

    def backward(self):
        pass


class Activation:
    def __init__(self):
        pass

    def forward(self, input):
        pass

    def backward(self, output):
        pass

    def update(self, learning_rate):
        pass


class Metric:
    def __init__(self):
        pass

    def update(self, output, target):

        pass

    def reset(self):
        pass

    def result(self):
        pass

