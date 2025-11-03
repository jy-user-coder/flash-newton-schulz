import torch


def reference_ns(G, steps=50, eps=1e-7):
    # Save original shape
    original_shape = G.shape

    # Handle higher dimensional tensors
    if G.ndim > 2:
        G = G.view(G.size(0), -1)

    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.clone()

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    # Handle tall matrices
    is_tall = X.size(0) > X.size(1)
    if is_tall:
        X = X.transpose(0, 1)

    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.transpose(0, 1)
        B = b * A + c * A @ A
        X = a * X + B @ X

    # Transpose back if needed
    if is_tall:
        X = X.transpose(0, 1)

    # Restore original shape
    if G.ndim > 2:
        X = X.view(original_shape)

    return X
