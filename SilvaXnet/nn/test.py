from Linear import Linear 
from Activation import * 



layer = Linear(10, 5)
x = cp.random.randn(32, 10)  # batch size 32, 10 features
output = layer.forward(x)
assert output.shape == (32, 5)

layer = Linear(10, 5)
x = cp.random.randn(32, 10)  # batch size 32, 10 features
output = layer.forward(x)
assert output.shape == (32, 5)

