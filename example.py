"""
nanoTorch Example Script - Fixed Version
Demonstrates all features of the nanoTorch library
"""

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

import numpy as np
import sys
import os

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nanotorch import Tensor, nn, optim, data, utils, functional as F
from nanotorch.tensor import cat, stack

def example_1_dnn_classification():
    """Dense Neural Network Classification"""
    print("\n1. Dense Neural Network Classification")
    print("-" * 80)
    
    # Create synthetic data
    X_train = Tensor(np.random.randn(1000, 20), device='cpu')
    y_train = np.random.randint(0, 5, 1000)
    
    # Create dataset and dataloader
    dataset = data.TensorDataset(X_train, Tensor(y_train))
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Build model
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 5)
    )
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.StepLR(optimizer, step_size=10, gamma=0.5)
    early_stopping = utils.EarlyStopping(patience=5, mode='min')
    
    # Training loop
    model.train()
    for epoch in range(20):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            # Forward pass
            pred = model(batch_X)
            loss = F.cross_entropy(pred, batch_y.data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.data
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/20, Loss: {avg_loss:.4f}, LR: {optimizer.lr:.6f}")
        
        if early_stopping(avg_loss):
            print("Early stopping triggered!")
            break
    
    # Evaluation
    model.eval()
    pred = model(X_train)
    acc = utils.accuracy(pred.numpy(), y_train)
    print(f"Training Accuracy: {acc:.4f}")
    print(f"Total Parameters: {utils.count_parameters(model):,}")
    
    return model, optimizer, avg_loss

def example_2_cnn():
    """Convolutional Neural Network"""
    print("\n2. Convolutional Neural Network")
    print("-" * 80)
    
    # Create a proper CNN model (without Sequential for now to avoid flattening issues)
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(32 * 8 * 8, 128)
            self.relu3 = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 10)
            
            self._modules = [self.conv1, self.conv2, self.fc1, self.fc2]
        
        def __call__(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)
            # Flatten
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
            x = self.fc1(x)
            x = self.relu3(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    cnn_model = SimpleCNN()
    print(f"CNN Parameters: {utils.count_parameters(cnn_model):,}")
    
    # Test forward pass
    X_images = Tensor(np.random.randn(8, 3, 32, 32), device='cpu')
    output = cnn_model(X_images)
    print(f"Input shape: {X_images.shape}, Output shape: {output.shape}")

def example_3_rnn():
    """RNN for Sequence Modeling"""
    print("\n3. RNN Recurrent Neural Network")
    print("-" * 80)
    
    # Sequence data (seq_len, batch, input_size)
    X_seq = Tensor(np.random.randn(10, 32, 50), device='cpu')
    
    rnn_model = nn.RNN(input_size=50, hidden_size=128, num_layers=2, device='cpu')
    output, hidden = rnn_model(X_seq)
    
    print(f"Input shape: {X_seq.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden state length: {len(hidden)}")
    print(f"RNN Parameters: {utils.count_parameters(rnn_model):,}")

def example_4_lstm():
    """LSTM for Sequence Modeling"""
    print("\n4. LSTM Recurrent Neural Network")
    print("-" * 80)
    
    # Sequence data (seq_len, batch, input_size)
    X_seq = Tensor(np.random.randn(10, 32, 50), device='cpu')
    
    lstm_model = nn.LSTM(input_size=50, hidden_size=128, num_layers=2, device='cpu')
    
    # LSTM returns (output, hidden_states, cell_states)
    result = lstm_model(X_seq)
    output = result[0]
    hidden = result[1]
    cell = result[2]
    
    print(f"Input shape: {X_seq.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Hidden states: {len(hidden)} layers")
    print(f"Cell states: {len(cell)} layers")
    print(f"LSTM Parameters: {utils.count_parameters(lstm_model):,}")

def example_5_gru():
    """GRU Network"""
    print("\n5. GRU Recurrent Neural Network")
    print("-" * 80)
    
    X_seq = Tensor(np.random.randn(10, 32, 50), device='cpu')
    
    gru_model = nn.GRU(input_size=50, hidden_size=128, num_layers=2, device='cpu')
    output, hidden = gru_model(X_seq)
    
    print(f"GRU Output shape: {output.shape}")
    print(f"GRU Hidden states: {len(hidden)} layers")
    print(f"GRU Parameters: {utils.count_parameters(gru_model):,}")

def example_6_transformer():
    """Transformer Architecture"""
    print("\n6. Transformer Encoder")
    print("-" * 80)
    
    # Transformer input (batch, seq_len, d_model)
    X_trans = Tensor(np.random.randn(16, 20, 512), device='cpu')
    
    transformer_layer = nn.TransformerEncoderLayer(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        device='cpu'
    )
    
    output = transformer_layer(X_trans)
    print(f"Transformer input: {X_trans.shape}")
    print(f"Transformer output: {output.shape}")
    print(f"Transformer Parameters: {utils.count_parameters(transformer_layer):,}")

def example_7_attention():
    """Multi-Head Attention"""
    print("\n7. Multi-Head Attention Mechanism")
    print("-" * 80)
    
    attention = nn.MultiHeadAttention(embed_dim=256, num_heads=8, device='cpu')
    query = Tensor(np.random.randn(16, 10, 256), device='cpu')
    key = Tensor(np.random.randn(16, 10, 256), device='cpu')
    value = Tensor(np.random.randn(16, 10, 256), device='cpu')
    
    attn_output = attention(query, key, value)
    print(f"Attention output shape: {attn_output.shape}")
    print(f"Attention Parameters: {utils.count_parameters(attention):,}")

def example_8_embedding():
    """Embedding Layer"""
    print("\n8. Embedding Layer for NLP")
    print("-" * 80)
    
    vocab_size = 10000
    embed_dim = 300
    embedding = nn.Embedding(vocab_size, embed_dim, device='cpu')
    
    # Token indices
    tokens = np.random.randint(0, vocab_size, (32, 20))  # batch=32, seq_len=20
    embedded = embedding(tokens)
    print(f"Token indices shape: {tokens.shape}")
    print(f"Embedded shape: {embedded.shape}")
    print(f"Embedding Parameters: {utils.count_parameters(embedding):,}")

def example_9_activations():
    """Various Activations"""
    print("\n9. Activation Functions")
    print("-" * 80)
    
    x_test = Tensor(np.linspace(-3, 3, 100), device='cpu')
    
    activations = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(0.2),
        'ELU': nn.ELU(),
        'GELU': nn.GELU(),
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh()
    }
    
    for name, activation in activations.items():
        output = activation(x_test)
        print(f"{name}: min={output.data.min():.4f}, max={output.data.max():.4f}")

def example_10_normalization():
    """Normalization Layers"""
    print("\n10. Normalization Layers")
    print("-" * 80)
    
    # Batch Normalization
    bn = nn.BatchNorm1d(64, device='cpu')
    x_bn = Tensor(np.random.randn(128, 64), device='cpu')
    x_normalized = bn(x_bn)
    print(f"BatchNorm1d: Input mean={x_bn.data.mean():.4f}, Output mean={x_normalized.data.mean():.4f}")
    
    # Layer Normalization
    ln = nn.LayerNorm(512, device='cpu')
    x_ln = Tensor(np.random.randn(32, 20, 512), device='cpu')
    x_norm = ln(x_ln)
    print(f"LayerNorm: Input std={x_ln.data.std():.4f}, Output std={x_norm.data.std():.4f}")

def example_11_optimizers():
    """Optimizer Comparison"""
    print("\n11. Optimizer Comparison")
    print("-" * 80)
    
    X_opt = Tensor(np.random.randn(100, 10), device='cpu')
    y_opt = Tensor(np.random.randn(100, 1), device='cpu')
    
    optimizers_test = {
        'SGD': (nn.Linear(10, 1, device='cpu'), optim.SGD),
        'Adam': (nn.Linear(10, 1, device='cpu'), optim.Adam),
        'AdamW': (nn.Linear(10, 1, device='cpu'), optim.AdamW),
        'RMSprop': (nn.Linear(10, 1, device='cpu'), optim.RMSprop)
    }
    
    for opt_name, (model, opt_class) in optimizers_test.items():
        optimizer = opt_class(model.parameters(), lr=0.01)
        pred = model(X_opt)
        loss = F.mse_loss(pred, y_opt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"{opt_name}: Loss = {loss.data:.4f}")

def example_12_schedulers():
    """Learning Rate Schedulers"""
    print("\n12. Learning Rate Schedulers")
    print("-" * 80)
    
    simple_model = nn.Linear(10, 1, device='cpu')
    
    schedulers = [
        ('StepLR', optim.Adam(simple_model.parameters(), lr=0.1), 
         lambda opt: optim.StepLR(opt, step_size=5, gamma=0.5)),
        ('CosineAnnealingLR', optim.Adam(simple_model.parameters(), lr=0.1),
         lambda opt: optim.CosineAnnealingLR(opt, T_max=10)),
        ('OneCycleLR', optim.Adam(simple_model.parameters(), lr=0.1),
         lambda opt: optim.OneCycleLR(opt, max_lr=0.5, total_steps=10))
    ]
    
    for sched_name, optimizer, sched_fn in schedulers:
        scheduler = sched_fn(optimizer)
        lrs = []
        for i in range(10):
            lrs.append(optimizer.lr)
            scheduler.step()
        print(f"{sched_name}: LR schedule = {[f'{lr:.4f}' for lr in lrs[:5]]}...")

def example_13_losses():
    """Loss Functions"""
    print("\n13. Loss Functions")
    print("-" * 80)
    
    pred_loss = Tensor(np.random.randn(32, 10), device='cpu')
    target_loss = np.random.randint(0, 10, 32)
    
    losses = {
        'CrossEntropy': F.cross_entropy(pred_loss, target_loss),
        'MSE': F.mse_loss(pred_loss, Tensor(np.random.randn(32, 10))),
        'MAE': F.mae_loss(pred_loss, Tensor(np.random.randn(32, 10))),
        'Huber': F.huber_loss(pred_loss, Tensor(np.random.randn(32, 10)))
    }
    
    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value.data:.4f}")

def example_14_checkpointing(model, optimizer, avg_loss):
    """Model Checkpoint Save/Load"""
    print("\n14. Model Checkpoint Save/Load")
    print("-" * 80)
    
    try:
        # Save checkpoint
        utils.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=20,
            filepath='model_checkpoint.pkl',
            loss=avg_loss
        )
        print("✓ Checkpoint saved to 'model_checkpoint.pkl'")
        
        # Load checkpoint
        checkpoint = utils.load_checkpoint('model_checkpoint.pkl', model=model, optimizer=optimizer)
        print(f"✓ Checkpoint loaded - Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    except Exception as e:
        print(f"⚠ Checkpoint test skipped: {e}")

def example_15_metrics():
    """Evaluation Metrics"""
    print("\n15. Evaluation Metrics")
    print("-" * 80)
    
    # Generate predictions
    predictions = np.random.rand(100, 5)
    targets = np.random.randint(0, 5, 100)
    
    acc = utils.accuracy(predictions, targets)
    precision, recall, f1 = utils.precision_recall_f1(predictions, targets, num_classes=5)
    cm = utils.confusion_matrix(predictions, targets, num_classes=5)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix shape: {cm.shape}")

def example_16_dataloader():
    """DataLoader Features"""
    print("\n16. DataLoader Features")
    print("-" * 80)
    
    dataset_dl = data.TensorDataset(
        Tensor(np.random.randn(1000, 10)),
        Tensor(np.random.randint(0, 3, 1000))
    )
    
    dataloader_dl = data.DataLoader(
        dataset_dl,
        batch_size=64,
        shuffle=True,
        drop_last=True
    )
    
    print(f"Dataset size: {len(dataset_dl)}")
    print(f"Number of batches: {len(dataloader_dl)}")
    
    batch_count = 0
    for batch_x, batch_y in dataloader_dl:
        batch_count += 1
        if batch_count == 1:
            print(f"First batch - X shape: {batch_x.shape}, y shape: {batch_y.shape}")
    
    print(f"Total batches processed: {batch_count}")

def example_17_grad_clipping():
    """Gradient Clipping"""
    print("\n17. Gradient Clipping")
    print("-" * 80)
    
    test_params = [Tensor(np.random.randn(10, 10), requires_grad=True, device='cpu') for _ in range(3)]
    for p in test_params:
        p.grad = p.xp.random.randn(*p.shape) * 10  # Large gradients
    
    grad_norm_before = sum((p.grad ** 2).sum() for p in test_params) ** 0.5
    utils.clip_grad_norm_(test_params, max_norm=1.0)
    grad_norm_after = sum((p.grad ** 2).sum() for p in test_params) ** 0.5
    
    print(f"Gradient norm before clipping: {grad_norm_before:.4f}")
    print(f"Gradient norm after clipping: {grad_norm_after:.4f}")

def example_18_tensor_ops():
    """Advanced Tensor Operations"""
    print("\n18. Advanced Tensor Operations")
    print("-" * 80)
    
    t1 = Tensor(np.random.randn(2, 3, 4), device='cpu')
    t2 = Tensor(np.random.randn(2, 3, 4), device='cpu')
    
    # Concatenate
    t_cat = cat([t1, t2], axis=0)
    print(f"Cat shape: {t_cat.shape}")
    
    # Stack
    t_stack = stack([t1, t2], axis=0)
    print(f"Stack shape: {t_stack.shape}")
    
    # Reshape
    t_reshape = t1.reshape(2, 12)
    print(f"Reshape: {t1.shape} -> {t_reshape.shape}")
    
    # Transpose
    t_transpose = t1.transpose(0, 2, 1)
    print(f"Transpose: {t1.shape} -> {t_transpose.shape}")
    
    # Squeeze/Unsqueeze
    t_unsqueeze = t1.unsqueeze(0)
    print(f"Unsqueeze: {t1.shape} -> {t_unsqueeze.shape}")
    
    t_squeeze = t_unsqueeze.squeeze(0)
    print(f"Squeeze: {t_unsqueeze.shape} -> {t_squeeze.shape}")

def print_summary():
    """Print library summary"""
    print("\n" + "=" * 80)
    print("🎉 nanoTorch Library Successfully Loaded!")
    print("=" * 80)
    print("\nAvailable Components:")
    print("  ✓ Tensor with Autograd")
    print("  ✓ Layers: Linear, Conv2d, RNN, LSTM, GRU, Embedding, Attention")
    print("  ✓ Activations: ReLU, LeakyReLU, ELU, GELU, Sigmoid, Tanh, Softmax")
    print("  ✓ Normalization: BatchNorm, LayerNorm, GroupNorm")
    print("  ✓ Pooling: MaxPool2d, AvgPool2d, AdaptiveAvgPool2d")
    print("  ✓ Regularization: Dropout, Dropout2d")
    print("  ✓ Optimizers: SGD, Adam, AdamW, RMSprop")
    print("  ✓ Schedulers: StepLR, CosineAnnealingLR, OneCycleLR")
    print("  ✓ Loss Functions: CrossEntropy, MSE, MAE, BCE, Huber")
    print("  ✓ Data: Dataset, TensorDataset, DataLoader")
    print("  ✓ Utils: Gradient clipping, Early stopping, Metrics, Checkpointing")
    print("  ✓ Architectures: Transformer, Multi-head Attention")
    print("  ✓ GPU Support: CuPy backend (if available)")
    
    if CUPY_AVAILABLE:
        print("\n✓ GPU acceleration available via CuPy!")
    else:
        print("\n⚠ GPU acceleration not available (CuPy not installed)")
    
    print("\n" + "=" * 80)
    print("Usage:")
    print("  from nanotorch import nn, optim, data, utils, functional as F")
    print("  ")
    print("  # Create model")
    print("  model = nn.Sequential(")
    print("      nn.Linear(784, 128),")
    print("      nn.ReLU(),")
    print("      nn.Dropout(0.2),")
    print("      nn.Linear(128, 10)")
    print("  )")
    print("  ")
    print("  # Train")
    print("  optimizer = optim.AdamW(model.parameters(), lr=0.001)")
    print("  loss = F.cross_entropy(pred, target)")
    print("  loss.backward()")
    print("  optimizer.step()")
    print("=" * 80)

if __name__ == "__main__":
    print("nanoTorch v0.1.0 - Production-Ready Deep Learning Library")
    print("=" * 80)
    
    # Run all examples
    model, optimizer, avg_loss = example_1_dnn_classification()
    example_2_cnn()
    example_3_rnn()
    example_4_lstm()
    example_5_gru()
    example_6_transformer()
    example_7_attention()
    example_8_embedding()
    example_9_activations()
    example_10_normalization()
    example_11_optimizers()
    example_12_schedulers()
    example_13_losses()
    example_14_checkpointing(model, optimizer, avg_loss)
    example_15_metrics()
    example_16_dataloader()
    example_17_grad_clipping()
    example_18_tensor_ops()
    
    print_summary()