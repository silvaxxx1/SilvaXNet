# SilvaXNet 🚀 (formerly SilvaNet)

Welcome to **SilvaXNet**, the next evolution of **SilvaNet**! Now, with built-in **GPU acceleration via CuPy**, SilvaXNet provides a seamless deep learning experience for both CPU and GPU users. Whether you're an AI enthusiast, researcher, or educator, this library offers a **lightweight, intuitive**, and **educational** deep learning framework that runs efficiently on **both CPU (NumPy) and GPU (CuPy)**.

<p align="center">
  <img src="silvaxnet.png" alt="Quantization Overview">
</p>

---

## 🚀 What's New?

At this moment, we have **SilvaNet** (CPU version), with the intent to extend to **SilvaXNet** (GPU version).

### ⚡ SilvaNet (CPU version, NumPy-based)
- Retains **pure NumPy** implementation for maximum portability
- Ideal for environments **without GPU support**

### 🔥 SilvaXNet (Upcoming GPU-accelerated version with CuPy)
- Planned **CuPy** integration for GPU acceleration
- Seamless NumPy ↔ CuPy tensor operations
- Optimized matrix operations for speedup

### ✅ Unified API for CPU & GPU (Future Feature)
- Effortless switching between **SilvaNet (NumPy)** and **SilvaXNet (CuPy)**
- API remains **consistent** for both backends

### 🔬 Planned Improvements for Neural Network Support
- **Convolutional Layers (CNNs) optimized for GPU**
- Better **gradient computation with autograd**
- Enhanced support for **ANNs, RNNs, LSTMs, GRUs, and more!**

---

## 🌟 Key Features (Current SilvaNet Version)

- **Autograd Support**: Automatic differentiation for smooth backpropagation.
- **Deep Learning Layers**:
  - Fully Connected (Dense) Layers
  - Recurrent Layers: **RNN, LSTM, GRU**
  - Convolutional Layers (**Currently CPU-based**)
- **Loss Functions**: Cross-Entropy, MSE, and more.
- **Optimized Computation**: NumPy-based operations for efficiency.
- **Model Management**: Save and load trained models seamlessly.

---

## 🚀 Getting Started

Here's a quick example using **SilvaNet** (CPU-only):

```python
import numpy as np  # For CPU (SilvaNet)
from nn.Layers import Sequential, Dense
from nn.losses import CrossEntropyLoss
from nn.optimizer import SGD
from Network import NeuralNetwork

# Sample data
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100, 1))

# Define the neural network
model = Sequential()
model.add(Dense(n_inputs=10, n_units=64, activation='relu'))
model.add(Dense(n_inputs=64, n_units=32, activation='relu'))
model.add(Dense(n_inputs=32, n_units=2))

# Loss function and optimizer
loss_fn = CrossEntropyLoss()
optimizer = SGD(parameters=model.get_parameters(), alpha=0.01)

# Train
nn = NeuralNetwork(model, loss_fn, optimizer)
nn.fit(X_train, y_train, epochs=100, batch_size=16)
```

---

## ⚙️ Installation

To install **SilvaNet**, clone the repository:

```bash
git clone https://github.com/silvaxxx1/SilvaXNet
cd SilvaXNet
```

For **SilvaNet (CPU)**:
```bash
pip install -r requirements_cpu.txt  # NumPy-based
```

GPU support (**SilvaXNet**) is currently in development.

---

## 📚 Documentation

Check out the **full API reference, guides, and tutorials** here: [Documentation](link-to-documentation)

---

## 🤝 Contributing

We welcome contributions from the community! If you want to improve **SilvaNet** or help develop **SilvaXNet**, check out our [contributing guidelines](link-to-contributing-guidelines). We’d love to hear your feedback!

---

## 📝 License

SilvaNet is licensed under the [MIT License](link-to-license). See the LICENSE file for more details.

---

