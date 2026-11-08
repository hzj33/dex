#!/usr/bin/env python3
"""Summarize student acceptance metrics from logs + TensorBoard event files.

Example:
  python scripts/summarize_student_acceptance.py \
    --run padapt=outputs/XHandHoraScrewDriver_student_padapt/run_a \
    --run purebc=outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min \
    --run diffusion_latent=outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_seed42_15min \
    --run diffusion_action_chunk=outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_fix_seed42_15min \
    --output docs/stage_acceptance_summary.md
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    HAS_TB = True
except Exception:
    HAS_TB = False


RE_BEST = re.compile(r"Current Best:\s*(-?\d+(?:\.\d+)?)")
RE_LAST_FPS = re.compile(r"Last FPS:\s*(-?\d+(?:\.\d+)?)")
RE_ERRORS = re.compile(
    r"(Traceback \(most recent call last\)|\bRuntimeError:\b|\bAssertionError:\b|CUDA error|"
    r"\bHydraException:\b|\bValueError:\b|\bKeyError:\b|\bnan\b|\bNaN\b)"
)


def parse_run_arg(s: str) -> Tuple[str, Path]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--run expects label=path, got: {s}")
    label, path = s.split("=", 1)
    label = label.strip()
    path = Path(path.strip())
    if not label:
        raise argparse.ArgumentTypeError(f"Empty label in --run: {s}")
    return label, path


def find_train_log(run_dir: Path) -> Optional[Path]:
    logs = sorted(
        run_dir.glob("train*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    return logs[-1] if logs else None


def parse_log_metrics(log_path: Optional[Path]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "max_current_best": None,
        "last_current_best": None,
        "median_last_fps": None,
        "error_hits": None,
    }
    if log_path is None or not log_path.exists():
        return out
    txt = log_path.read_text(errors="ignore")
    best_vals = [float(x) for x in RE_BEST.findall(txt)]
    fps_vals = [float(x) for x in RE_LAST_FPS.findall(txt)]
    err_hits = len(RE_ERRORS.findall(txt))
    if best_vals:
        out["max_current_best"] = max(best_vals)
        out["last_current_best"] = best_vals[-1]
    if fps_vals:
        s = sorted(fps_vals)
        n = len(s)
        out["median_last_fps"] = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    out["error_hits"] = float(err_hits)
    return out


def find_latest_event_file(run_dir: Path) -> Optional[Path]:
    event_files = sorted(
        run_dir.rglob("events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime,
    )
    return event_files[-1] if event_files else None


def parse_tb_last_values(event_file: Optional[Path], wanted_tags: List[str]) -> Dict[str, Optional[float]]:
    out = {k: None for k in wanted_tags}
    if not HAS_TB or event_file is None:
        return out
    try:
        acc = EventAccumulator(str(event_file))
        acc.Reload()
        scalar_tags = set(acc.Tags().get("scalars", []))
        for tag in wanted_tags:
            if tag in scalar_tags:
                vals = acc.Scalars(tag)
                if vals:
                    out[tag] = float(vals[-1].value)
    except Exception:
        return out
    return out


def find_best_ckpt(run_dir: Path) -> Optional[Path]:
    cands = sorted(run_dir.rglob("model_best.ckpt"))
    return cands[-1] if cands else None


def fmt(v: Optional[float], nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    return f"{v:.{nd}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="label=run_dir")
    parser.add_argument(
        "--output",
        default="docs/stage_acceptance_summary.md",
        help="output markdown path",
    )
    args = parser.parse_args()

    run_items = [parse_run_arg(x) for x in args.run]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wanted_tags = [
        "episode_rewards/step",
        "episode_lengths/step",
        "total_loss/frame",
        "latent_loss/frame",
        "bc_loss/frame",
        "diffusion_loss/frame",
        "first_action_bc_loss/frame",
        "chunk_bc_loss/frame",
        "done_rate/frame",
    ]

    rows = []
    for label, run_dir in run_items:
        log_path = find_train_log(run_dir)
        log_metrics = parse_log_metrics(log_path)
        event_file = find_latest_event_file(run_dir)
        tb_metrics = parse_tb_last_values(event_file, wanted_tags)
        best_ckpt = find_best_ckpt(run_dir)
        rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "log_path": str(log_path) if log_path else "N/A",
                "event_file": str(event_file) if event_file else "N/A",
                "best_ckpt": str(best_ckpt) if best_ckpt else "N/A",
                **log_metrics,
                **tb_metrics,
            }
        )

    lines: List[str] = []
    lines.append("# Student Acceptance Summary")
    lines.append("")
    lines.append("## Unified Metrics Table")
    lines.append("")
    lines.append(
        "| Algorithm | Max Current Best | Last EpReward | Last EpLen | Last DoneRate | Last TotalLoss | Median LastFPS | Error Hits | Best CKPT |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['label']} | {fmt(r['max_current_best'])} | {fmt(r['episode_rewards/step'])} | "
            f"{fmt(r['episode_lengths/step'])} | {fmt(r['done_rate/frame'])} | {fmt(r['total_loss/frame'])} | "
            f"{fmt(r['median_last_fps'])} | {fmt(r['error_hits'], nd=0)} | `{r['best_ckpt']}` |"
        )

    lines.append("")
    lines.append("## Run Artifacts")
    lines.append("")
    for r in rows:
        lines.append(f"- `{r['label']}`")
        lines.append(f"  - run_dir: `{r['run_dir']}`")
        lines.append(f"  - log: `{r['log_path']}`")
        lines.append(f"  - event: `{r['event_file']}`")
        lines.append(f"  - best_ckpt: `{r['best_ckpt']}`")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `Current Best` comes from training stdout parsing, aligned with existing acceptance scripts.")
    lines.append("- TensorBoard metrics are read from the latest event file under each run directory.")
    lines.append("- If `done_rate/frame` or env-derived metrics are `N/A`, that run likely predates metric instrumentation.")

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote summary: {output_path}")


if __name__ == "__main__":
    main()
