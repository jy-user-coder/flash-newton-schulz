import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from ns_variants.flash_ns import (
    newton_schulz_torch,
    newton_schulz_triton_dion,
    newton_schulz_triton_aol,
)
from ns_variants import polar_express, reference_ns

# --- config ---
ns_impls = [
    "aol", 
    "muon",
    # "dion", 
    "std_pe", 
    "aol_pe",
]
ns_impl = {
    "aol": newton_schulz_triton_aol,
    "dion": newton_schulz_triton_dion,
    "muon": newton_schulz_torch,
    "std_pe": partial(polar_express, aol=False),
    "aol_pe": partial(polar_express, aol=True),
}
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- compute errors per step and impl ---
rows = []

for impl in ns_impls:
    dirpath = f"grads/{impl}"
    filenames_grads = [f for f in os.listdir(dirpath) if f.endswith(".pth")]
    print(f"Found files: {filenames_grads}")
    loss_vals = pd.read_csv(os.path.join(dirpath, "loss_vals.csv"))
    loss_map = dict(zip(loss_vals["step"], loss_vals["val_loss"]))

    by_step = defaultdict(list)
    for fn in filenames_grads:
        # grads/ns_impl/update_size0_size1_step.pth
        step = int(os.path.splitext(fn)[0].split("_")[-1])
        by_step[step].append(fn)

    for step, fns in sorted(by_step.items()):
        errs = []
        for fn in fns:
            grad = torch.load(os.path.join(dirpath, fn), map_location=device)

            with torch.no_grad():
                ref = reference_ns(grad)
                approx = ns_impl[impl](grad)

                num = torch.linalg.norm(approx - ref, ord="fro", dim=(-2, -1))
                den = torch.linalg.norm(ref, ord="fro", dim=(-2, -1))
                eps = torch.finfo(ref.dtype).eps if ref.is_floating_point() else 1e-12
                ratio = num / den.clamp_min(eps)
                if ratio.ndim > 0:
                    ratio = ratio.mean()
                errs.append(float(ratio))

        rows.append(
            {
                "step": step,
                "val_loss": float(loss_map.get(step, float("nan"))),
                "impl": impl,
                "polar_error": float(sum(errs) / len(errs)) if errs else float("nan"),
            }
        )

df = pd.DataFrame(rows)

# drop step == 0 entirely
df = df[df["step"] != 0].sort_values(["impl", "step"])

# --- plot: steps vs polar_error, color=val_loss (log), marker by impl ---
markers = {"aol": "o", "dion": "s", "std_pe": "^", "aol_pe": "D"}
plt.figure(figsize=(8, 5))

vals = df["val_loss"].to_numpy(dtype=float)
pos = vals[np.isfinite(vals) & (vals > 0)]
vmin = float(pos.min()) if pos.size else 1e-12
vmax = float(pos.max()) if pos.size else 1.0
norm = LogNorm(vmin=vmin, vmax=vmax)

last = None
for impl in ns_impls:
    d = df[df.impl == impl]
    if d.empty:
        continue
    last = plt.scatter(
        d.step,
        d.polar_error,
        c=d.val_loss,
        marker=markers[impl],
        norm=norm,
        label=impl,
    )

cbar = plt.colorbar(last)
cbar.set_label("val_loss (log)")

# Legend with same color for all shapes
legend_handles = [
    Line2D([0], [0], marker=markers[i], linestyle="", markerfacecolor="black", markeredgecolor="black", label=i)
    for i in ns_impls
]
plt.legend(handles=legend_handles, title="ns_impl")

plt.xlabel("steps")
plt.ylabel("polar error")
plt.tight_layout()
plt.savefig("figs/polar_error_training.png")
plt.close()

