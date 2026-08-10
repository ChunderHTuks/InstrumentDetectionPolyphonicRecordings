import numpy as np
import torch
import torch.nn as nn

def bce_with_logits_loss_from_scratch(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    term1 = np.maximum(logits, 0)
    term2 = -logits * targets
    term3 = np.log(1 + np.exp(-np.abs(logits)))

    elementwise_loss = term1 + term2 + term3

    return np.mean(elementwise_loss)

logits_np = np.array([[
    1.5, -0.8, 2.1, -3.0, 0.1, 0.5, -1.2, 4.0, -2.5, 0.9, 1.1
]], dtype=np.float64)

targets_np = np.array([[
    1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1
]], dtype=np.float64)

loss_from_scratch = bce_with_logits_loss_from_scratch(logits_np, targets_np)
logits_torch = torch.from_numpy(logits_np)
targets_torch = torch.from_numpy(targets_np)

criterion = nn.BCEWithLogitsLoss()

loss_pytorch = criterion(logits_torch, targets_torch)

print(f"Input Logits Shape: {logits_np.shape}")
print(f"Input Targets Shape: {targets_np.shape}")
print("-" * 40)
print(f"Logits:\n{logits_np}")
print(f"Targets:\n{targets_np}")
print("-" * 40)
print(f"Loss from custom implementation: {loss_from_scratch:.8f}")
print(f"Loss from PyTorch:         {loss_pytorch.item():.8f}")
print("-" * 40)

is_verified = np.allclose(loss_from_scratch, loss_pytorch.item())
print(f"Verification successful: {is_verified}")