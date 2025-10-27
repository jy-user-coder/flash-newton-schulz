import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
print("Running script. Figures will be saved to ./figs")

import torch


@torch.compile(dynamic=False, fullgraph=True)
def newton_schulz_torch(G, epsilon=1e-7):
    """
    Reference implementation of Newton-Schulz without Triton.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def aol(X):
    A = X.mT @ X
    s = torch.rsqrt(torch.clamp_min(A.abs().sum(dim=-1, keepdim=False), min=1e-10))
    X = X * s.unsqueeze(-1)
    return X


def get_error(size, bias=0.0):
    X = torch.randn((size, size)) + bias
    device = "cuda"
    X = X.to(device)
    aol_X = aol(X)
    U, S, V = torch.linalg.svd(X)
    aol_U, aol_S, aol_V = torch.linalg.svd(aol_X)
    UVh = newton_schulz_torch(X)
    diff_aol_to_svd = torch.linalg.norm(aol_U @ aol_V - U @ V, ord=2).item()
    diff_ns_to_svd = torch.linalg.norm(UVh - U @ V, ord=2).item()
    return diff_aol_to_svd, diff_ns_to_svd


import matplotlib.pyplot as plt

sizes = [8, 12, 32, 64, 128, 512, 1024]
errors = [get_error(size) for size in sizes]
plt.plot(
    sizes,
    [error[0] for error in errors],
    label=r"$\| U'V'^T - U V^T\|_2$",
)
plt.legend()

import matplotlib.pyplot as plt
import torch


def get_error_low_rank(size, rank, device="cuda"):
    """
    Generates a matrix of a specific rank to show the failure of quasi-orthogonality.
    The error should remain large regardless of size for rank < size.
    """
    # 1. Generate a low-rank matrix.
    if rank >= size:
        X_raw = torch.randn((size, size))
        X = (1.0 / torch.sqrt(torch.tensor(float(size)))) * X_raw
    else:
        # Generate two smaller matrices
        A = torch.randn((size, rank))
        B = torch.randn((rank, size))
        # Their product is a low-rank matrix
        X = A @ B

        # --- Normalization (Important for Fair Comparison) ---
        X = X / torch.linalg.norm(X, ord="fro") * torch.sqrt(torch.tensor(float(size)))
    X = X.to(device)
    aol_X = aol(X)

    # Calculate polar decompositions
    U, S, Vh = torch.linalg.svd(X)  # Vh is already V.T
    aol_U, aol_S, aol_Vh = torch.linalg.svd(aol_X)

    UVh = U @ Vh
    aol_UVh = aol_U @ aol_Vh

    # Compute the relative Frobenius norm error
    norm_UVh = torch.sqrt(torch.tensor(float(size)))
    diff_aol_to_svd = torch.linalg.norm(aol_UVh - UVh, ord="fro").item() / norm_UVh

    return diff_aol_to_svd


# Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

num_avgs = 3
plt.figure(figsize=(10, 6), dpi=200)

# Define the ranks you want to test
ranks_to_test = [1, 20, "full"]

for rank in ranks_to_test:
    sizes = [32, 64]
    # For low ranks, we can't create a matrix smaller than the rank
    sizes = [s for s in sizes if rank == "full" or s >= rank]

    errors = [0 for _ in sizes]
    for i in range(num_avgs):
        print(f"Running average {i+1}/{num_avgs} for rank {rank}...")
        current_run_errors = [
            get_error_low_rank(size, rank if rank != "full" else size, device=device)
            for size in sizes
        ]
        errors = [e + cre / num_avgs for e, cre in zip(errors, current_run_errors)]

    label_str = f"Rank = {rank}" if rank != "full" else "Full Rank (Quasi-Orthogonal)"
    plt.plot(sizes, errors, label=label_str, marker="o", markersize=4)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Matrix Dimension (n x n)")
plt.ylabel("Relative Polar Error (Frobenius Norm)")
plt.title("Polar Factor Perturbation vs. Matrix Rank")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.legend()
plt.savefig("figs/polar_error_rank.png")

import torch


def aol(X):
    coefs = torch.rsqrt((torch.abs(X.mT @ X)).sum(dim=1))
    return X @ torch.diag(coefs)


@torch.compile(dynamic=False, fullgraph=True)
def newton_schulz_torch(G, epsilon=1e-7):
    """
    Reference implementation of Newton-Schulz without Triton.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)
    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def aol_muon(X):
    return newton_schulz_torch(aol(X))


def objective(X):
    U, S, Vh = torch.linalg.svd(X)
    GT = U @ Vh
    approx = aol_muon(X)
    num = torch.linalg.norm(approx - GT, ord="fro")
    den = torch.linalg.norm(GT, ord="fro")
    return num / (den + 1e-8)


A = torch.randn((32, 32), device=device, requires_grad=True)
matrices, steps, loss_vals = [], [], []
for i in range(10000):
    loss = objective(A)
    loss.backward()
    with torch.no_grad():
        A += 0.1 * A.grad
    A.grad.zero_()
    if i % 200 == 0:
        matrices.append(A.cpu().detach().clone())
        steps.append(i)
    loss_vals.append(loss.item())
    print(f"loss: {loss.item()}", end="\r", flush=True)


import matplotlib.pyplot as plt

with torch.no_grad():
    for step, matrix in zip(steps, matrices):
        plt.figure()
        U, S, Vh = torch.linalg.svd(matrix)
        plt.hist(S.detach().cpu().numpy(), bins=20)
        plt.title(f"SVs at step:{step}")

plt.figure()
plt.plot(loss_vals)
plt.title("Evolution of polar error")

import io
import imageio.v3 as iio
import numpy as np
from matplotlib.colors import Normalize
from IPython.display import Image

frames = []
with torch.no_grad():
    for step, matrix in zip(steps, matrices):
        plt.figure(figsize=(6, 4), dpi=120)
        S = sorted(torch.linalg.svdvals(matrix).detach().cpu().numpy())
        plt.plot(S)
        plt.title(f"SVs at step:{step}")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        frames.append(iio.imread(buf.read()))
        plt.close()

iio.imwrite("figs/sv_hist.gif", np.stack(frames, axis=0), duration=0.5, loop=0)
print("wrote sv_hist.gif")

Image(filename="figs/sv_hist.gif")

fig, axes = plt.subplots(1, 2, layout="constrained")

AhA = (A.mT @ A).detach().cpu().numpy()
AhA = AhA / AhA.max()
B = torch.randn((32, 32))
BhB = (B.mT @ B).detach().cpu().numpy()
BhB = BhB / BhB.max()

norm = Normalize(vmin=min(AhA.min(), BhB.min()), vmax=max(AhA.max(), BhB.max()))

im1 = axes[0].imshow(AhA, norm=norm)
axes[0].axis("off")

im2 = axes[1].imshow(BhB, norm=norm)
axes[1].axis("off")

fig.colorbar(im1, ax=axes, location="right")
plt.savefig("figs/dist_ortho.png")
