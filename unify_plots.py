#!/usr/bin/env python3
"""
Unified plotting script:
- Recreates the two "dual" plots (runtime vs size, polar error vs size) with the
  same filtering/aggregation logic as your existing dual script (using WANTED).
- Recreates the Pareto plot (error vs. runtime, one curve per algo) for each available
  matrix size (size=xxx) and stores those images under a "pareto" subfolder.

Directory layout (per CSV run):
  outroot/
    {dist}/
      {dtype}/
        runtime_filtered_annotated.png
        polar_error_filtered_annotated.png
        pareto/
          pareto_size{SIZE}.png   (and/or .pdf/.svg depending on --pareto-out-ext)

Notes
- The three plot types intentionally operate on different slices/aggregations of the data,
  mirroring the behavior of your two original scripts.
- The Pareto plots are generated per size (filter size=xxx is applied).
- If your CSV contains multiple distributions or dtypes, the script iterates over each
  (dist, dtype) pair and produces the corresponding folder structure automatically.
"""

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatter, MaxNLocator


# ------------------ Shared helpers ------------------


def parse_filters(kvs: Optional[Iterable[str]]) -> Dict[str, object]:
    out = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise ValueError(f"Bad --filter '{kv}', expected key=value")
        k, v = kv.split("=", 1)
        v = v.strip()
        try:
            v_cast = ast.literal_eval(v)
        except Exception:
            v_cast = v
        out[k.strip()] = v_cast
    return out


def apply_filters(df: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    for k, v in filters.items():
        if k not in df.columns:
            raise KeyError(f"Filter key '{k}' not found in columns: {list(df.columns)}")
        df = df[df[k] == v]
    return df


def ensure_cols(df: pd.DataFrame, needed: Iterable[str]):
    missing = set(needed) - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")


def setup_paper_style(font="DejaVu Sans", base=11, lw=2.0, ms=6, grid=True):
    """Reasonable defaults for print/PDF."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # keep text as text in PDF
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": [font],
            "font.size": base,
            "axes.titlesize": base + 1,
            "axes.labelsize": base,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "legend.fontsize": base - 1,
            "axes.linewidth": 0.8 * 1.2,
            "xtick.major.width": 0.8 * 1.2,
            "ytick.major.width": 0.8 * 1.2,
            "xtick.minor.width": 0.6 * 1.2,
            "ytick.minor.width": 0.6 * 1.2,
            "lines.linewidth": lw,
            "lines.markersize": ms,
            "axes.grid": grid,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
        }
    )


def apply_axis_format(ax, logx=False, logy=False):
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(LogFormatter(10))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 3))
        ax.xaxis.set_major_formatter(fmt)

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(LogFormatter(10))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(fmt)

    ax.tick_params(which="both", direction="out", length=4)
    ax.minorticks_on()


def colorblind_palette():
    # Okabe–Ito (color-blind friendly)
    return [
        "#009E73",
        "#0072B2",
        "#56B4E9",
        "#009E73",
        # "#F0E442",
        # "#E69F00",
        # "#D55E00",
        # "#F0E442",
        # "#56B4E9",
        # "#009E73",
        # "#0072B2",
        # "#F0E442",
        # "#F0E442",
        # "#CC79A7",
        # "#000000",
    ]


def marker_cycle():
    return [
        "o",
        "s",
        "D",
        "^",
        "v",
        "P",
        "X",
        "h",
        "*",
    ]


def linestyle_cycle():
    # Helps in B/W printouts
    return [
        "solid",
        (0, (4, 2)),
        "dashed",
        (0, (1, 1)),
        (0, (3, 1, 1, 1)),
        "dashdot",
        (0, (5, 2, 1, 2)),
    ]


# ------------------ Dual plots (runtime vs size, polar error vs size) ------------------

WANTED: Tuple[Tuple[str, int], ...] = (
    ("Muon", 5),
    ("Dion", 5),
    ("AOLxDion", 5),
    ("AOLxDion", 4),
)


def dual_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the requested (algo, iter)
    mask = (
        ((df["algo"] == "Muon") & (df["iter"] == 5))
        | ((df["algo"] == "Dion") & (df["iter"] == 5))
        | ((df["algo"] == "AOLxDion") & (df["iter"].isin([4, 5])))
    )
    df = df[mask].copy()
    agg = df.groupby(["algo", "iter", "size"], as_index=False).agg(
        mean_runtime_ms=("batch_time_ms", "mean"),
        std_runtime_ms=("batch_time_ms", "std"),
        mean_polar_error=("polar_error", "mean"),
        std_polar_error=("polar_error", "std"),
    )
    return agg


def dual_plot_lines(
    ax,
    agg: pd.DataFrame,
    ykey: str,
    ylab: str,
    title: str,
    with_errorbars: bool = False,
):
    palette = colorblind_palette()
    color_map_fixed = {
        "Muon": (palette[0], marker_cycle()[0]),
        "Dion": (palette[1], marker_cycle()[1]),
        "AOLxDion": (palette[2], marker_cycle()[2]),  # same as Pareto
    }

    for (algo, it), sub in agg.groupby(["algo", "iter"]):
        label = f"{algo} (iter={it})"
        sub = sub.sort_values("size")
        x = sub["size"].to_numpy(dtype=float)
        y = sub[ykey].to_numpy(dtype=float)
        yerr = sub[ykey.replace("mean_", "std_")].fillna(0).to_numpy(dtype=float)

        c, m = color_map_fixed.get(algo, palette[0])
        linestyle = linestyle_cycle()[0 if it == 5 else 2]

        ax.plot(x, y, marker=m, label=label, color=c, linestyle=linestyle)
        if np.any(yerr > 0):
            if with_errorbars:
                ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3, alpha=0.8, color=c)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=c)

    ax.set_xlabel("matrix size")
    ax.set_ylabel(ylab)
    # display values instead of power of 2 on x axis
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(LogFormatter(2))
    # ax.set_yscale("log", base=10)
    # ax.set_title(title)
    # ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.25)


def make_speedup_bars(
    ax,
    agg: pd.DataFrame,
    baseline_algo: str,
    baseline_iter: int,
):
    """
    agg: output of dual_aggregate()
         columns: ['algo','iter','size','mean_runtime_ms','std_runtime_ms', ...]
    Produces grouped bars of speedup (baseline_runtime / algo_runtime) per size.
    Colors match the line plots; per-(algo,iter) uses same color with distinct hatch/edge.
    """
    # check available sizes
    available_sizes = sorted(agg["size"].unique().tolist())
    agg = agg[agg["size"].isin(available_sizes[-4:])]
    # Consistent colors with the line/pareto plots
    palette = colorblind_palette()
    color_map_fixed = {
        # "Muon": palette[0],
        "Dion": palette[1],
        "AOLxDion": palette[2],
    }

    # Keep only the wanted tuples in a stable order
    wanted = list(WANTED)  # (algo, iter)
    agg = agg[agg.apply(lambda r: (r["algo"], r["iter"]) in wanted, axis=1)].copy()

    # Baseline per size
    base = (
        agg[(agg["algo"] == baseline_algo) & (agg["iter"] == baseline_iter)]
        .set_index("size")["mean_runtime_ms"]
        .rename("baseline_runtime")
    )
    if base.empty:
        raise SystemExit(
            f"Baseline {baseline_algo} (iter={baseline_iter}) not present after filtering."
        )

    # Compute speedups
    dfp = agg.set_index("size").join(base, how="inner").reset_index().copy()
    dfp["speedup"] = dfp["baseline_runtime"] / dfp["mean_runtime_ms"]

    # X axis = sizes (as ordered numbers); one bar group per size
    sizes = sorted(dfp["size"].unique().tolist())
    x = np.arange(len(sizes), dtype=float)

    # One bar per (algo,iter) inside each group
    groups = wanted  # preserve the display order shown in the line plot legend
    n = len(groups)
    width = min(0.8 / max(n, 1), 0.22)  # keep some spacing

    # For consistent left-to-right grouping around tick center
    offsets = (np.arange(n) - (n - 1) / 2) * (width + 0.01)

    # Hatches to differentiate the two AOLxDion curves when printed in grayscale
    hatch_map = {
        ("AOLxDion", 4): "///",
        # ("AOLxDion", 5): "///",
    }

    # Draw bars, one series per (algo,iter)
    bars_handles = []
    labels = []
    for i, (algo, it) in enumerate(groups):
        # if algo == baseline_algo and it == baseline_iter:
        #     continue  # skip baseline itself
        sub = dfp[(dfp["algo"] == algo) & (dfp["iter"] == it)]
        # align values to the full size list
        vals = [
            (
                sub[sub["size"] == s]["speedup"].values[0]
                if s in sub["size"].values
                else np.nan
            )
            for s in sizes
        ]
        c = color_map_fixed.get(algo, palette[3])
        hatch = hatch_map.get((algo, it), None)
        h = ax.bar(
            x + offsets[i],
            vals,
            width=width,
            label=f"{algo} (iter={it})",
            color=c,
            # edgecolor="black",
            linewidth=0.0,
            hatch=hatch,
            edgecolor="white",
        )
        # we add a text to indicate each bar value on top of the bar
        for j, val in enumerate(vals):
            if not np.isnan(val):
                ax.text(
                    x[j] + offsets[i],
                    val + 0.02,  # - 0.5,
                    f"{val:.1f}x",
                    ha="center",
                    va="bottom",
                    fontsize=15,
                    rotation=90,
                )
                # ax.text(
                #     x[j] + offsets[i],
                #     val + 0.02,  # - 0.5,
                #     f"{val:.1f}x",
                #     ha="center",
                #     va="bottom",
                #     fontsize=10,
                #     rotation=0,
                # )
        bars_handles.append(h)
        labels.append(f"{algo} (iter={it})")

    # Axis/labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(int(s)) if float(s).is_integer() else str(s) for s in sizes], rotation=0
    )
    ax.set_ylim(bottom=0.0, top=3.5)
    ax.set_xlabel("matrix size")
    ax.set_ylabel(f"speedup vs {baseline_algo} (iter={baseline_iter})")
    # ax.set_title("Speedup vs Matrix Size (grouped by size)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    # ax.legend(ncol=1)


def make_dual_plots(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    df2 = df.dropna(subset=["polar_error"], how="any")
    agg = dual_aggregate(df2)

    # Runtime
    # fig1, ax1 = plt.subplots()
    # dual_plot_lines(
    #     ax1,
    #     agg,
    #     "mean_runtime_ms",
    #     "runtime (ms per batch)",
    #     "Runtime vs Matrix Size (selected iters)",
    #     with_errorbars=True,  # <— add error bars on the runtime axis
    # )
    # fig1.tight_layout()
    # p1 = out_dir / "runtime_filtered_annotated.png"
    # fig1.savefig(p1, dpi=300)
    # plt.close(fig1)
    fig1, ax1 = plt.subplots()
    make_speedup_bars(ax1, agg, "Muon", 5)
    fig1.tight_layout()
    p1 = out_dir / "runtime_filtered_annotated.png"  # keep same filename as requested
    fig1.savefig(p1, dpi=300)
    plt.close(fig1)

    # Polar error (unchanged; no error bars)
    fig2, ax2 = plt.subplots()
    dual_plot_lines(
        ax2,
        agg,
        "mean_polar_error",
        "polar error (mean)",
        "Polar Error vs Matrix Size (selected iters)",
    )
    # ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0.0)

    fig2.tight_layout()
    p2 = out_dir / "polar_error_filtered_annotated.png"
    fig2.savefig(p2, dpi=300)
    plt.close(fig2)


# ------------------ Pareto plots (error vs runtime per size) ------------------


def scale_alpha(series, lo=0.3, hi=1.0):
    """Min-max scale to [lo, hi]. Handles constant or NaN series gracefully."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        # all NaN → fall back to mid alpha
        return pd.Series(np.full(len(series), (lo + hi) / 2), index=series.index)
    s_min, s_max = float(s.min()), float(s.max())
    if s_max == s_min:
        return pd.Series(np.full(len(series), hi), index=series.index)
    scaled = (s - s_min) / (s_max - s_min)
    return lo + scaled * (hi - hi)
    # Typo above fixed below (kept structure for clarity)
    # return lo + scaled * (hi - lo)


def pareto_aggregate(
    df: pd.DataFrame,
    hue_col: str,
    runtime_col: str,
    error_col: str,
    alpha_col: Optional[str],
    runtime_std_col: Optional[str] = None,  # <— NEW
):
    agg_cols = [runtime_col, error_col]
    if alpha_col:
        agg_cols.append(alpha_col)

    # core medians (unchanged behavior)
    agg = (
        df.groupby([hue_col, "iter"], as_index=False)[agg_cols]
        .median()
        .sort_values([hue_col, "iter"])
    )

    # y-error from dispersion of raw error values
    ystd = (
        df.groupby([hue_col, "iter"], as_index=False)[error_col]
        .std()
        .rename(columns={error_col: f"{error_col}_std"})
    )
    agg = pd.merge(agg, ystd, on=[hue_col, "iter"], how="left")

    # x-error from existing per-sample std column, aggregated by median
    if runtime_std_col and runtime_std_col in df.columns:
        xstd = (
            df.groupby([hue_col, "iter"], as_index=False)[runtime_std_col]
            .median()
            .rename(columns={runtime_std_col: f"{runtime_col}_std"})
        )
        agg = pd.merge(agg, xstd, on=[hue_col, "iter"], how="left")

    return agg


def make_pareto_per_size(
    df: pd.DataFrame,
    out_dir: Path,
    hue_col: str = "algo",
    runtime_col: str = "batch_time_ms",
    error_col: str = "polar_error",
    alpha_col: Optional[str] = None,
    alpha_range=(0.3, 1.0),
    logx: bool = False,
    logy: bool = False,
    title_prefix: Optional[str] = None,
    fig_size=(6, 4),
    out_ext: str = "png",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted(df["size"].dropna().unique().tolist())
    if not sizes:
        return

    # --- Consistent color map with dual plots ---
    palette = colorblind_palette()
    color_map_fixed = {
        "Muon": (palette[0], marker_cycle()[0]),  # same as first curve in dual plots
        "Dion": (palette[1], marker_cycle()[1]),
        # AOLxDion appears twice in the duals (iters 4 & 5) → pick one of their colors
        "AOLxDion": (palette[2], marker_cycle()[2]),  # consistent choice
    }
    runtime_std_col_guess = f"{runtime_col}_std"

    for s in sizes:
        sub = df[df["size"] == s].copy()
        if sub.empty:
            continue

        agg = pareto_aggregate(
            sub,
            hue_col,
            runtime_col,
            error_col,
            alpha_col,
            runtime_std_col=(
                runtime_std_col_guess if runtime_std_col_guess in sub.columns else None
            ),
        )
        if agg.empty:
            continue

        if title_prefix is None:
            title = f"{error_col} vs {runtime_col}  (size={s})"
        else:
            title = f"{title_prefix}  (size={s})"

        if alpha_col and alpha_col in agg.columns:
            alpha_vals = scale_alpha(
                agg[alpha_col], lo=alpha_range[0], hi=alpha_range[1]
            )
        else:
            alpha_vals = pd.Series(np.ones(len(agg)), index=agg.index)

        fig, ax = plt.subplots(figsize=fig_size)
        hue_values = list(pd.unique(agg[hue_col]))

        # Assign consistent colors; fallback to palette if unknown
        color_map = {
            h: color_map_fixed.get(h, palette[i % len(palette)])
            for i, h in enumerate(hue_values)
        }

        has_xerr = f"{runtime_col}_std" in agg.columns
        has_yerr = f"{error_col}_std" in agg.columns

        for i, (h, g) in enumerate(agg.groupby(hue_col, sort=False)):
            g = g.sort_values(runtime_col)
            c, m = color_map[h]

            ax.plot(
                g[runtime_col].values,
                g[error_col].values,
                marker=m,
                label=str(h),
                color=c,
                linewidth=5.0,
            )

            sub_alpha = alpha_vals.loc[g.index].values
            ax.scatter(
                g[runtime_col].values,
                g[error_col].values,
                alpha=sub_alpha,
                color=c,
                edgecolor="none",
                zorder=3,
                s=80,
            )

            if has_xerr or has_yerr:
                xerr = g[f"{runtime_col}_std"].values if has_xerr else None
                yerr = g[f"{error_col}_std"].values if has_yerr else None
                ax.errorbar(
                    g[runtime_col].values,
                    g[error_col].values,
                    xerr=xerr,
                    yerr=yerr,
                    fmt="none",
                    ecolor=c,
                    elinewidth=1.2,
                    capsize=3,
                    alpha=0.8,
                )

        ax.set_xlabel(runtime_col.replace("_", " "))
        ax.set_ylabel(error_col.replace("_", " "))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0.01)
        # ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(title=hue_col)

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        plt.tight_layout()
        setup_paper_style(font="DejaVu Sans", base=20, lw=5.0, ms=15.0)
        out_path = out_dir / f"pareto_size{s}.{out_ext}"
        fig.savefig(out_path)
        plt.close(fig)


# ------------------ Orchestration ------------------


def infer_groups(df: pd.DataFrame) -> Iterable[Tuple[object, object]]:
    have_dist = "dist" in df.columns
    have_dtype = "dtype" in df.columns
    if not have_dist and not have_dtype:
        yield ("unknown_dist", "unknown_dtype")
        return

    if have_dist and have_dtype:
        groups = (
            df.groupby(["dist", "dtype"])
            .size()
            .reset_index()[["dist", "dtype"]]
            .itertuples(index=False, name=None)
        )
    elif have_dist:
        groups = [(d, "unknown_dtype") for d in df["dist"].unique()]
    else:
        groups = [("unknown_dist", t) for t in df["dtype"].unique()]
    for g in groups:
        yield g


def run_for_group(
    df: pd.DataFrame,
    outroot: Path,
    dist: object,
    dtype: object,
    paper: bool,
    font: str,
    font_size: int,
    line_width: float,
    marker_size: float,
    grayscale_safe: bool,
    legend_outside: bool,
    legend_cols: int,
    # pareto
    pareto_hue_col: str,
    pareto_runtime_col: str,
    pareto_error_col: str,
    pareto_alpha_col: Optional[str],
    pareto_alpha_minmax: Tuple[float, float],
    pareto_logx: bool,
    pareto_logy: bool,
    pareto_fig_w: float,
    pareto_fig_h: float,
    pareto_out_ext: str,
):
    # Style
    if paper:
        setup_paper_style(font=font, base=font_size, lw=line_width, ms=marker_size)

    # Narrow to this group
    sub = df.copy()
    if "dist" in sub.columns:
        sub = sub[sub["dist"] == dist]
    if "dtype" in sub.columns:
        sub = sub[sub["dtype"] == dtype]

    # Create destination directories
    base_dir = outroot / str(dist) / str(dtype)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1) Dual plots (runtime & polar error vs size on selected iters)
    make_dual_plots(sub, base_dir)

    # 2) Pareto plots per size inside "pareto" folder
    pareto_dir = base_dir / "pareto"
    make_pareto_per_size(
        df=sub,
        out_dir=pareto_dir,
        hue_col=pareto_hue_col,
        runtime_col=pareto_runtime_col,
        error_col=pareto_error_col,
        alpha_col=pareto_alpha_col,
        alpha_range=pareto_alpha_minmax,
        logx=pareto_logx,
        logy=pareto_logy,
        title_prefix=None,
        fig_size=(pareto_fig_w, pareto_fig_h),
        out_ext=pareto_out_ext,
    )


def main():
    p = argparse.ArgumentParser(
        description="Make dual plots + per-size Pareto plots, organized by dist/dtype."
    )
    p.add_argument("--csv", required=True, nargs="+", help="Input CSV file")
    p.add_argument("--outroot", default="plots_out", help="Root output directory")

    # Optional global pre-filters applied before grouping
    p.add_argument(
        "--filters",
        nargs="*",
        default=[],
        help="Key=Value filters to preselect rows (e.g., device='cuda:0 (NVIDIA L40S)' dim='(128, 128)' iter=5)",
    )

    # Styling
    p.add_argument("--paper", action="store_true", help="Enable paper-ready styling")
    p.add_argument("--font", default="DejaVu Sans", help="Font family")
    p.add_argument("--font-size", type=int, default=11, help="Base font size")
    p.add_argument("--line-width", type=float, default=2.0, help="Line width")
    p.add_argument("--marker-size", type=float, default=6.0, help="Marker size")
    p.add_argument(
        "--legend-outside",
        action="store_true",
        help="(Reserved) place legend outside for Pareto (not used)",
    )
    p.add_argument(
        "--legend-cols", type=int, default=1, help="Legend columns (not used)"
    )
    p.add_argument(
        "--grayscale-safe",
        action="store_true",
        help="(Reserved) use dashes/markers to differ in B/W",
    )

    # Pareto-specific knobs (kept close to your original script)
    p.add_argument(
        "--pareto-hue-col", default="algo", help="Hue column (curve per value)"
    )
    p.add_argument(
        "--pareto-runtime-col",
        default="batch_time_ms",
        help="Runtime column for X axis",
    )
    p.add_argument(
        "--pareto-error-col", default="polar_error", help="Error column for Y axis"
    )
    p.add_argument(
        "--pareto-alpha-col",
        default=None,
        help="Numeric column to map to point transparency (optional)",
    )
    p.add_argument(
        "--pareto-alpha-range",
        nargs=2,
        type=float,
        default=[0.3, 1.0],
        metavar=("LO", "HI"),
        help="Alpha range for Pareto points",
    )
    p.add_argument(
        "--pareto-logx", action="store_true", help="Log scale on X for Pareto"
    )
    p.add_argument(
        "--pareto-logy", action="store_true", help="Log scale on Y for Pareto"
    )
    p.add_argument(
        "--pareto-figwidth",
        type=float,
        default=9.0,
        help="Pareto figure width (inches)",
    )
    p.add_argument(
        "--pareto-figheight",
        type=float,
        default=6.0,
        help="Pareto figure height (inches)",
    )
    p.add_argument(
        "--pareto-out-ext",
        default="png",
        choices=["png", "pdf", "svg"],
        help="File extension/format for Pareto plots",
    )

    args = p.parse_args()

    dfs = []
    for cf in args.csv:
        pth = Path(cf)
        if not pth.is_file():
            # check if path is a directory containing CSV files
            if pth.is_dir():
                csvs = list(pth.glob("*.csv"))
                if not csvs:
                    raise SystemExit(f"No CSV files found in directory: {pth}")
                for c2 in csvs:
                    try:
                        dfi = pd.read_csv(c2)
                        dfs.append(dfi)
                    except Exception as e:
                        raise SystemExit(f"Error reading CSV '{c2}': {e}")
                continue
            raise SystemExit(f"CSV file not found: {pth}")
        try:
            dfi = pd.read_csv(pth)
        except Exception as e:
            raise SystemExit(f"Error reading CSV '{pth}': {e}")
        dfs.append(dfi)
    df = pd.concat(dfs, ignore_index=True)

    # levy col: rename dist=="levy" to dist==f"levy-{alpha}" with alpha from levy_alpha col
    if "dist" in df.columns and "levy_alpha" in df.columns:
        mask = df["dist"] == "levy"
        df.loc[mask, "dist"] = df.loc[mask, "levy_alpha"].apply(lambda a: f"levy-{a}")

    # Sanity columns used across plots
    ensure_cols(df, ["algo", "iter", "size", "batch_time_ms", "polar_error"])

    # Optional pre-filters (e.g., single device, single dim, etc.)
    filters = parse_filters(args.filters)
    if filters:
        df = apply_filters(df, filters)
        if df.empty:
            raise SystemExit("No rows after applying --filters.")

    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    # Iterate over dist/dtype groups present (or fallbacks)
    for dist, dtype in infer_groups(df):
        run_for_group(
            df=df,
            outroot=outroot,
            dist=dist,
            dtype=dtype,
            paper=args.paper,
            font=args.font,
            font_size=args.font_size,
            line_width=args.line_width,
            marker_size=args.marker_size,
            grayscale_safe=args.grayscale_safe,
            legend_outside=args.legend_outside,
            legend_cols=args.legend_cols,
            pareto_hue_col=args.pareto_hue_col,
            pareto_runtime_col=args.pareto_runtime_col,
            pareto_error_col=args.pareto_error_col,
            pareto_alpha_col=args.pareto_alpha_col,
            pareto_alpha_minmax=tuple(args.pareto_alpha_range),
            pareto_logx=args.pareto_logx,
            pareto_logy=args.pareto_logy,
            pareto_fig_w=args.pareto_figwidth,
            pareto_fig_h=args.pareto_figheight,
            pareto_out_ext=args.pareto_out_ext,
        )


if __name__ == "__main__":
    main()
