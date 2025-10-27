import os, math, argparse, numpy as np, torch, matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from tqdm import tqdm

from gpt import GPTConfig, GPT


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.model_max_length = 10**9
    return tok


def build_model():
    cfg = GPTConfig(
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
    )
    model = GPT(cfg)
    return model


def load_text():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw = load_dataset("text", data_files={"train": url})
    text = "\n".join(raw["train"]["text"])
    return text


# ----------------- dataset -----------------
class BlocksDS(Dataset):
    def __init__(self, text, tok, block=128, stride=64):
        ids = tok(text)["input_ids"]
        # sliding windows
        self.samples = [
            ids[i : i + block]
            for i in range(0, len(ids) - 1, stride)
            if len(ids[i : i + block]) > 1
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return {"input_ids": self.samples[i]}


# ----------------- helpers -----------------
def one_batch(ds, bs, device, collator):
    loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collator)
    batch = next(iter(loader))
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def _l2(x):
    return torch.linalg.norm(x, ord=2)


def _prep_grad_matrix(grad: torch.Tensor) -> torch.Tensor:
    """
    grad: [out, in] like Linear.weight.grad
    Make it tall: rows >= cols
    Normalize by global L2
    """
    if grad.size(0) < grad.size(1):
        grad = grad.T
    g = grad / _l2(grad)
    return g


def collect_stats_for_batch(
    model,
    batch,
    topk_svs=64,
    max_side=128,
):
    """
    Returns:
      dict_svs[name]          -> np.array[topk_svs]
      dict_gtg_small[name]    -> np.array[<=max_side, <=max_side] float16
      dict_si_small[name]     -> np.array[<=max_side] float16
    All on CPU. Keeps memory bounded.
    """
    model.train()
    model.zero_grad(set_to_none=True)

    out, loss = model(idx=batch["input_ids"], targets=batch["labels"])
    loss.backward()

    dict_svs = {}
    dict_gtg_small = {}
    dict_si_small = {}

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if p.grad.ndim != 2:
            continue

        # build normalized tall grad matrix g
        g = _prep_grad_matrix(p.grad.detach())

        # singular values
        svs = torch.linalg.svdvals(g)
        if topk_svs is not None and svs.numel() > topk_svs:
            svs = svs[:topk_svs]
        dict_svs[name] = svs.detach().cpu().numpy()

        # Gram = g^T g which is square (cols x cols)
        gtg = g.T @ g  # [d, d]
        si = torch.rsqrt(gtg.abs().sum(1))  # stability-like coeff vector

        # spatial downsample so we never keep a 3000x3000 matrix
        side = gtg.shape[0]
        stride = max(1, side // max_side)
        gtg_small = gtg[::stride, ::stride].to(torch.float16).cpu().numpy()
        si_small = si[::stride].to(torch.float16).cpu().numpy()

        dict_gtg_small[name] = gtg_small
        dict_si_small[name] = si_small

    # free grads on GPU
    model.zero_grad(set_to_none=True)

    return dict_svs, dict_gtg_small, dict_si_small


# ----------------- plotting utils -----------------
def safe_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def plot_svs_by_bs(dict_svs_by_bs, outpath):
    """
    dict_svs_by_bs:
      {name: {bs: svs_array}}
    Make a grid of line plots. Cheap in memory.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    names = sorted(dict_svs_by_bs.keys())
    n = len(names)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        for bs in sorted(dict_svs_by_bs[name].keys()):
            s = dict_svs_by_bs[name][bs]
            x = np.arange(1, len(s) + 1)
            ax.plot(x, np.sort(s)[::-1], label=f"bs={bs}")
        ttl = name if len(name) <= 40 else name[:37] + "..."
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("SV index")
        ax.set_ylabel("SV")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
    # delete any unused axes
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j // ncols][j % ncols])

    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close(fig)


def plot_gtg_by_step(dict_gtg_small_by_step, steps_all, outdir):
    """
    For each parameter matrix name:
      make one figure with len(steps_all) subplots in a row
      each subplot is a heatmap of downsampled gtg at that step
    Memory stays bounded because we create and close per-name figures.
    """
    os.makedirs(outdir, exist_ok=True)

    for name in sorted(dict_gtg_small_by_step.keys()):
        steps_for_name = [st for st in steps_all if st in dict_gtg_small_by_step[name]]
        if len(steps_for_name) == 0:
            continue

        fig, axes = plt.subplots(
            1,
            len(steps_for_name),
            figsize=(4 * len(steps_for_name), 3.2),
            squeeze=False,
        )
        axes = axes[0]

        for j, st in enumerate(steps_for_name):
            ax = axes[j]
            g_small = dict_gtg_small_by_step[name][st]
            im = ax.imshow(g_small, interpolation="nearest", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"step {st}", fontsize=8)

        fig.suptitle(name, fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f"gtg_steps_{safe_name(name)}.png"),
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)


def plot_si_over_steps(dict_si_by_step, outdir):
    """
    For each parameter matrix name:
      make one figure overlaying si curves for each step.
      si arrays are already downsampled.
    """
    os.makedirs(outdir, exist_ok=True)

    for name in sorted(dict_si_by_step.keys()):
        steps_for_name = sorted(dict_si_by_step[name].keys())
        if len(steps_for_name) == 0:
            continue

        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for st in steps_for_name:
            si_small = dict_si_by_step[name][st]
            # sort for monotonic display
            ax.plot(np.sort(si_small), label=f"step {st}", linewidth=1)

        ax.set_title(name if len(name) <= 60 else name[:57] + "...", fontsize=9)
        ax.set_xlabel("index (sorted)")
        ax.set_ylabel("si")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f"coeffs_steps_{safe_name(name)}.png"),
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)


def plot_svs_over_steps(dict_svs_by_step, outdir):
    """
    For completeness: overlay top-k singular values per step for each matrix.
    """
    os.makedirs(outdir, exist_ok=True)

    for name in sorted(dict_svs_by_step.keys()):
        steps_for_name = sorted(dict_svs_by_step[name].keys())
        if len(steps_for_name) == 0:
            continue

        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for st in steps_for_name:
            s = dict_svs_by_step[name][st]
            x = np.arange(1, len(s) + 1)
            ax.plot(x, np.sort(s)[::-1], label=f"step {st}", linewidth=1)

        ax.set_title(name if len(name) <= 60 else name[:57] + "...", fontsize=9)
        ax.set_xlabel("SV index")
        ax.set_ylabel("SV")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f"svs_steps_{safe_name(name)}.png"),
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)


# ----------------- main routine -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of training steps",
    )
    parser.add_argument(
        "--eval_bs_list",
        type=str,
        default="3,5,10,20,40",
        help="Comma-separated batch sizes for BS analysis",
    )
    parser.add_argument(
        "--steps_eval_list",
        type=str,
        default="0,25,50,100",
        help="Comma-separated training steps for checkpoint stats",
    )
    parser.add_argument(
        "--fixed_eval_bs",
        type=int,
        default=32,
        help="Batch size for step-wise stats",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=128,
        help="Downsampled GTG heatmap max side length",
    )
    parser.add_argument(
        "--topk_svs",
        type=int,
        default=64,
        help="Top-k singular values to store per matrix",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("figs", exist_ok=True)

    tok = build_tokenizer()
    text = load_text()
    ds = BlocksDS(text, tok)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    model = build_model().to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collator)

    # Prepare evaluation checkpoints
    steps_eval = sorted({int(s) for s in args.steps_eval_list.split(",")})
    max_needed = max(steps_eval + [args.num_steps])
    _ = max_needed  # silence unused warning

    # Storage
    dict_svs_by_bs = {}  # {name: {bs: svs_vec}}
    dict_svs_by_step = {}  # {name: {step: svs_vec}}
    dict_gtg_small_by_step = {}  # {name: {step: gtg_small_mat}}
    dict_si_by_step = {}  # {name: {step: si_small_vec}}

    # Step 0 stats if requested
    if 0 in steps_eval:
        batch_fixed = one_batch(ds, args.fixed_eval_bs, device, collator)
        svs0, gtg0, si0 = collect_stats_for_batch(
            model,
            batch_fixed,
            topk_svs=args.topk_svs,
            max_side=args.max_side,
        )
        for name, arr in svs0.items():
            dict_svs_by_step.setdefault(name, {})[0] = arr
        for name, arr in gtg0.items():
            dict_gtg_small_by_step.setdefault(name, {})[0] = arr
        for name, arr in si0.items():
            dict_si_by_step.setdefault(name, {})[0] = arr

    # Train loop
    for step, batch in enumerate(loader, start=1):
        if step > args.num_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        out, loss = model(idx=batch["input_ids"], targets=batch["labels"])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        print(f"Step {step}/{args.num_steps}, loss: {loss.item():.4f}")

        if step in steps_eval:
            batch_fixed = one_batch(ds, args.fixed_eval_bs, device, collator)
            svs_ckpt, gtg_ckpt, si_ckpt = collect_stats_for_batch(
                model,
                batch_fixed,
                topk_svs=args.topk_svs,
                max_side=args.max_side,
            )

            for name, arr in svs_ckpt.items():
                dict_svs_by_step.setdefault(name, {})[step] = arr
            for name, arr in gtg_ckpt.items():
                dict_gtg_small_by_step.setdefault(name, {})[step] = arr
            for name, arr in si_ckpt.items():
                dict_si_by_step.setdefault(name, {})[step] = arr

    # Batch size sweep (SVs only, cheap)
    print("Collecting SVs for different batch sizes...")
    for bs in tqdm([int(x) for x in args.eval_bs_list.split(",")]):
        model.train()
        batch = one_batch(ds, bs, device, collator)
        svs_bs, _, _ = collect_stats_for_batch(
            model,
            batch,
            topk_svs=args.topk_svs,
            max_side=args.max_side,
        )
        for name, arr in svs_bs.items():
            dict_svs_by_bs.setdefault(name, {})[bs] = arr

    # ----------------- plots -----------------
    print("Plotting singular values vs batch size...")
    plot_svs_by_bs(dict_svs_by_bs, outpath="figs/svs_bs.png")

    print("Plotting downsampled G^T G heatmaps per layer across steps...")
    steps_all = sorted({st for d in dict_gtg_small_by_step.values() for st in d})
    plot_gtg_by_step(dict_gtg_small_by_step, steps_all, outdir="figs/gtg")

    print("Plotting coeff curves across steps...")
    plot_si_over_steps(dict_si_by_step, outdir="figs/aol_coeffs")

    print("Plotting SV curves across steps...")
    plot_svs_over_steps(dict_svs_by_step, outdir="figs/svs_layers")

    print("Done.")


if __name__ == "__main__":
    main()
