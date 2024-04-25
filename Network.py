from autograd import Tensor
import numpy as np
import pickle
from tqdm import tqdm
from tabulate import tabulate

class NeuralNetwork:
    def __init__(self, model, loss_fn, optimizer):
        """
        Initialize the NeuralNetwork class.

        Args:
            model (Sequential): The neural network model.
            loss_fn (LossFunction): The loss function for optimization.
            optimizer (Optimizer): The optimizer for updating model parameters.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def compile(self, loss_fn, optimizer):
        """
        Compile the neural network with the given loss function and optimizer.

        Args:
            loss_fn (LossFunction): The loss function for optimization.
            optimizer (Optimizer): The optimizer for updating model parameters.
        """
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def fit(self, X_train, y_train, epochs, batch_size):
        """
        Train the neural network on the given training data.

        Args:
            X_train (array-like): The input training data.
            y_train (array-like): The target training data.
            epochs (int): The number of epochs for training.
            batch_size (int): The batch size for training.
        """
        # Total number of batches per epoch
        total_batches_per_epoch = len(X_train) // batch_size

        # Create a single progress bar for the entire training process
        pbar = tqdm(total=total_batches_per_epoch * epochs, desc='Training', unit='batch', dynamic_ncols=True, mininterval=1)

        # Training loop
        for epoch in range(1, epochs + 1):  # Start from 1 instead of 0
            epoch_loss = 0
            for i in range(0, len(X_train), batch_size):
                X_batch = Tensor(X_train[i:i+batch_size], autograd=True)
                y_batch = Tensor(y_train[i:i+batch_size].astype(int), autograd=True)

                # Forward pass
                outputs = self.model.forward(X_batch)

                # Compute loss
                loss = self.loss_fn.forward(outputs, y_batch)
                epoch_loss += loss.data

                # Backward pass
                loss.backward()

                # Update parameters
                self.optimizer.step()

                pbar.update(1)  # Update the progress bar for each batch

            pbar.set_postfix(loss=epoch_loss / len(X_train))

            # Update the description to include current epoch
            pbar.set_description(f'Training: {epoch}/{epochs} epochs')

            # Check if we have completed 100 epochs and reset the progress bar
            if epoch % 100 == 0 or epoch == epochs:
                pbar.close()
                if epoch < epochs:
                    pbar = tqdm(total=total_batches_per_epoch * epochs, desc=f'Training: {epoch}/{epochs} epochs',
                                unit='batch', dynamic_ncols=True, mininterval=1)

        # Close the progress bar at the end
        pbar.close()

    def evaluate(self, X_test, y_test):
        """
        Evaluate the neural network on the given test data.

        Args:
            X_test (array-like): The input test data.
            y_test (array-like): The target test data.
        """
        correct = 0
        for i in range(len(X_test)):
            X_sample = Tensor(X_test[i], autograd=True).reshape(1, -1)  # Reshape using view
            y_sample = y_test[i]

            # Forward pass
            outputs = self.model.forward(X_sample)

            # Calculate accuracy
            prediction = np.argmax(outputs.data)
            if prediction == y_sample:
                correct += 1

        accuracy = correct / len(X_test)
        print(f"Test Accuracy: {accuracy}")

    def predict(self, X):
        """
        Predict the output labels for the given input data.

        Args:
            X (array-like): The input data.

        Returns:
            numpy.ndarray: The predicted output labels.
        """
        predictions = []
        for x in X:
            X_sample = Tensor(x, autograd=True).reshape(1, -1)  # Reshape using view

            # Forward pass
            outputs = self.model.forward(X_sample)

            # Calculate prediction
            prediction = np.argmax(outputs.data)
            predictions.append(prediction)
        
        return np.array(predictions)

    def summary(self):
        """Print a summary of the neural network architecture."""
        headers = ["Layer", "Layer Type", "Output Shape", "Param #"]
        rows = []

        total_params = 0

        for i, layer in enumerate(self.model.layers):
            layer_type = layer.__class__.__name__

            # Get output shape if available, otherwise set to "N/A"
            output_shape = getattr(layer, "output_shape", "N/A")
            if output_shape != "N/A":
                output_shape = output_shape()

            # Convert Tensor objects to numpy arrays for shape calculation
            parameters = [param.data if isinstance(param, Tensor) else param for param in layer.parameters]

            param_count = sum([np.prod(param.shape) for param in parameters])

            total_params += param_count

            rows.append([i+1, layer_type, output_shape, param_count])

        # Add a row for the total number of parameters
        rows.append(["Total", "", "", total_params])

        print(tabulate(rows, headers=headers))

    def get_parameters(self):
        """Get the parameters of the neural network."""
        parameters = []
        for layer in self.model.layers:
            parameters.extend(layer.get_parameters())
        return parameters

    def save_model(self, filename):
        """
        Save the neural network model to a file.

        Args:
            filename (str): The filename to save the model.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_model(filename):
        """
        Load the neural network model from a file.

        Args:
            filename (str): The filename to load the model from.
        
        Returns:
            NeuralNetwork: The loaded neural network model.
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return NeuralNetwork(model, None, None)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the loaded neural network model on the given test data.

        Args:
            X_test (array-like): The input test data.
            y_test (array-like): The target test data.
        """
        correct = 0
        for i in range(len(X_test)):
            X_sample = Tensor(X_test[i], autograd=True).reshape(1, -1)  # Reshape using view
            y_sample = y_test[i]

            # Forward pass
            outputs = self.model.forward(X_sample)

            # Calculate accuracy
            prediction = np.argmax(outputs.data)
            if prediction == y_sample:
                correct += 1

        accuracy = correct / len(X_test)
        print(f"Test Accuracy: {accuracy}")

# Example usage:
# Create a neural network object
# model = NeuralNetwork(...)

# Train the model
# model.fit(X_train, y_train, epochs, batch_size)

# Evaluate the model
# model.evaluate(X_test, y_test)

# Predict using the model
# predictions = model.predict(X_test)

# Save the model
# model.save_model("model.pkl")

# Load the model
# loaded_model = NeuralNetwork.load_model("model.pkl")

# Evaluate the loaded model
# loaded_model.evaluate_model(X_test, y_test)
