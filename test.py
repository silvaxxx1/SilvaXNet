from autograd import Tensor
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from nn.Layers import Sequential, Dense
from nn.losses import CrossEntropyLoss
from nn.optimizer import SGD
from Network import NeuralNetwork

# Load Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
X_train = X_train / np.max(X_train, axis=0)
X_test = X_test / np.max(X_test, axis=0)

# Define the neural network architecture
model = Sequential()
model.add(Dense(n_inputs=X_train.shape[1], n_units=64, activation='relu'))
model.add(Dense(n_inputs=64, n_units=32, activation='relu'))
model.add(Dense(n_inputs=32, n_units=2))  # Output layer without activation for binary classification

# Define loss function and optimizer
loss_fn = CrossEntropyLoss()
optimizer = SGD(parameters=model.get_parameters(), alpha=0.01)

# Create a neural network instance
nn = NeuralNetwork(model, loss_fn, optimizer)

# Compile the model
nn.compile(loss_fn, optimizer)

# Print model summary
nn.summary()

# Train the model
nn.fit(X_train, y_train, epochs=100, batch_size=16)

# Evaluate the model
nn.evaluate(X_test, y_test)
