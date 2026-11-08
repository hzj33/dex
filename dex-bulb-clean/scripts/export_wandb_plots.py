#!/usr/bin/env python3
"""
Export high-resolution plots from a Weights & Biases run.

Examples:
  python scripts/export_wandb_plots.py \
    --run entity/project/run_id \
    --outdir outputs/wandb_plots/run_id \
    --dpi 300 \
    --smoothing 0.9 \
    --pdf

If you are syncing TensorBoard (sync_tensorboard=True), the x-axis is usually `_step`
and equals your TensorBoard step (e.g., agent_steps).
"""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple


def _parse_run_path(run: str) -> str:
    parts = [p for p in run.split("/") if p]
    if len(parts) != 3:
        raise ValueError(f"Invalid --run '{run}'. Expected: entity/project/run_id")
    return "/".join(parts)


def _safe_filename(name: str) -> str:
    name = name.strip().replace("/", "__")
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)
    return name[:200] if len(name) > 200 else name


def _ema(values: List[float], smoothing: float) -> List[float]:
    if not values:
        return []
    if smoothing <= 0.0:
        return list(values)
    if smoothing >= 1.0:
        return [values[0]] * len(values)
    out: List[float] = []
    last = values[0]
    out.append(last)
    for v in values[1:]:
        last = smoothing * last + (1.0 - smoothing) * v
        out.append(last)
    return out


def _downsample_xy(x: List[float], y: List[float], max_points: int) -> Tuple[List[float], List[float]]:
    if max_points <= 0 or len(x) <= max_points:
        return x, y
    stride = max(1, len(x) // max_points)
    xs = x[::stride]
    ys = y[::stride]
    if xs[-1] != x[-1]:
        xs.append(x[-1])
        ys.append(y[-1])
    return xs, ys


def _collect_series(
    history_rows: Iterable[Dict],
    metrics: List[str],
    x_key: str,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    xs: Dict[str, List[float]] = {m: [] for m in metrics}
    ys: Dict[str, List[float]] = {m: [] for m in metrics}
    for row in history_rows:
        x_val = row.get(x_key, None)
        if x_val is None:
            continue
        try:
            x_val_f = float(x_val)
        except Exception:
            continue

        for m in metrics:
            v = row.get(m, None)
            if v is None:
                continue
            try:
                v_f = float(v)
            except Exception:
                continue
            if math.isnan(v_f) or math.isinf(v_f):
                continue
            xs[m].append(x_val_f)
            ys[m].append(v_f)
    return xs, ys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="W&B run path: entity/project/run_id")
    parser.add_argument("--outdir", required=True, help="Output directory for PNG/PDF")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[
            "episode_rewards/step",
            "episode_lengths/step",
            "adapt_loss/frame",
            "wasserstein_dist/frame",
            "dual_loss/frame",
            "q_loss/frame",
            "policy_loss/frame",
            "gradient_penalty/frame",
        ],
        help="Metric keys to export (space-separated).",
    )
    parser.add_argument(
        "--x",
        default="_step",
        help="X-axis key in W&B history (default: _step).",
    )
    parser.add_argument("--smoothing", type=float, default=0.9, help="EMA smoothing in [0,1].")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (default: 300).")
    parser.add_argument("--max_points", type=int, default=6000, help="Downsample if too many points.")
    parser.add_argument("--pdf", action="store_true", help="Also write a multi-page PDF report.")
    parser.add_argument("--width", type=float, default=10.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=4.0, help="Figure height in inches.")
    args = parser.parse_args()

    run_path = _parse_run_path(args.run)
    os.makedirs(args.outdir, exist_ok=True)

    # Headless matplotlib backend.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    # scan_history returns an iterator over full history (slower but complete).
    keys = sorted(set([args.x] + list(args.metrics)))
    rows = run.scan_history(keys=keys)
    xs, ys = _collect_series(rows, args.metrics, args.x)

    pdf_path = os.path.join(args.outdir, "report.pdf")
    pdf = PdfPages(pdf_path) if args.pdf else None

    exported = 0
    for metric in args.metrics:
        x = xs.get(metric, [])
        y = ys.get(metric, [])
        if not x:
            print(f"[skip] no data for metric: {metric}")
            continue

        # Sort by x to avoid resume/restart disorder.
        pairs = sorted(zip(x, y), key=lambda p: p[0])
        x_sorted = [p[0] for p in pairs]
        y_sorted = [p[1] for p in pairs]
        x_plot, y_plot = _downsample_xy(x_sorted, y_sorted, args.max_points)
        y_smooth = _ema(y_plot, args.smoothing)

        fig, ax = plt.subplots(figsize=(args.width, args.height))
        ax.plot(x_plot, y_plot, linewidth=1.0, alpha=0.25, label="raw")
        ax.plot(x_plot, y_smooth, linewidth=2.0, label=f"ema({args.smoothing:g})")
        ax.set_title(f"{run_path} — {metric}")
        ax.set_xlabel(args.x)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        out_png = os.path.join(args.outdir, f"{_safe_filename(metric)}.png")
        fig.savefig(out_png, dpi=args.dpi)
        if pdf is not None:
            pdf.savefig(fig)
        plt.close(fig)
        exported += 1

        print(f"[ok] {metric} -> {out_png}")

    if pdf is not None:
        pdf.close()
        print(f"[ok] pdf -> {pdf_path}")

    if exported == 0:
        print("No plots exported. Check run path / metrics / permissions.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

