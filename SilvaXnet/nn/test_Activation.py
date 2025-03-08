import cupy as cp
import pytest
from Activation import Relu, Sigmoid, Tanh, LeakyRelu  # Correct the import names according to the new module structure

@pytest.mark.parametrize("activation_class", [Relu, Sigmoid, Tanh, LeakyRelu])
def test_forward(activation_class):
    # Create test data
    x = cp.array([[1.0, -1.0, 0.5], [-0.5, 2.0, -1.5]])

    # Initialize the activation function, passing default leaky_slope if needed
    if activation_class == LeakyRelu:
        activation = activation_class(leaky_slope=0.01)  # Pass default leaky_slope
    else:
        activation = activation_class()

    # Perform forward pass
    output = activation.forward(x)

    # Check if the output matches the expected values for each activation function
    if activation_class == Relu:
        expected_output = cp.maximum(0, x)
    elif activation_class == Sigmoid:
        expected_output = 1.0 / (1.0 + cp.exp(-x))
    elif activation_class == Tanh:
        expected_output = cp.tanh(x)
    elif activation_class == LeakyRelu:
        leaky_slope = 0.01  # Use the same default slope as in initialization
        expected_output = cp.maximum(leaky_slope * x, x)
    
    # Assert if the output is correct
    cp.testing.assert_allclose(output, expected_output, atol=1e-6)


@pytest.mark.parametrize("activation_class", [Relu, Sigmoid, Tanh, LeakyRelu])
def test_backward(activation_class):
    # Create test data for input and gradient
    x = cp.array([[1.0, -1.0, 0.5], [-0.5, 2.0, -1.5]])
    grad_output = cp.array([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]])

    # Initialize the activation function, passing default leaky_slope if needed
    if activation_class == LeakyRelu:
        activation = activation_class(leaky_slope=0.01)  # Pass default leaky_slope
    else:
        activation = activation_class()

    # Perform forward pass
    activation.forward(x)

    # Perform backward pass
    grad_input = activation.backward(grad_output)

    # Check if the backward pass produces the expected gradients
    if activation_class == Relu:
        grad_expected = grad_output * (x > 0)
    elif activation_class == Sigmoid:
        a = 1.0 / (1.0 + cp.exp(-x))
        grad_expected = grad_output * a * (1 - a)
    elif activation_class == Tanh:
        a = cp.tanh(x)
        grad_expected = grad_output * (1 - a**2)
    elif activation_class == LeakyRelu:
        grad_expected = grad_output * (x > 0) + grad_output * (x <= 0) * 0.01

    # Assert if the backward gradient is correct
    cp.testing.assert_allclose(grad_input, grad_expected, atol=1e-6)


def test_leaky_relu_slope():
    # Test Leaky ReLU with different slope values
    x = cp.array([[1.0, -1.0, 0.5], [-0.5, 2.0, -1.5]])

    # Test with different leaky slopes
    leaky_slopes = [0.01, 0.1, 0.2]
    for slope in leaky_slopes:
        activation = LeakyRelu(leaky_slope=slope)
        output = activation.forward(x)

        # Check that the output is correct with the given slope
        expected_output = cp.maximum(slope * x, x)
        cp.testing.assert_allclose(output, expected_output, atol=1e-6)


# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()
