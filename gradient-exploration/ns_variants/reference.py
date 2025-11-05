import torch

def reference_ns(G, steps=200, eps=1e-7):
    # Save original shape
    original_shape = G.shape

    a, b, c = (1.5, -0.5, 0.0) # For guaranteed convergence
    X = G.clone()

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    # Handle tall matrices
    is_tall = X.size(-2) > X.size(-1)
    if is_tall:
        X = X.mT

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    # Transpose back if needed
    if is_tall:
        X = X.mT

    return X
