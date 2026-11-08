#!/usr/bin/env python3
"""Run fixed rollout evals over checkpoint candidates and select deploy best.

Example:
  python scripts/eval_select_checkpoints.py \
    --task Dexh13HoraLightbulbSim2RealTwoFingerCoDrive \
    --algo PPO \
    --checkpoints 'outputs/run/stage1_nn/best_reward_*.pth' \
    --checkpoints 'outputs/run/stage1_nn/best_eval.pth' \
    --out-dir outputs/run/eval_select/deploy_eval
"""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import time
import math
from pathlib import Path


EVAL_RE = re.compile(
    r"EvalSummary\s+steps=(?P<steps>\d+)\s+"
    r"avg_reward=(?P<reward>[-+0-9.eE]+)\s+"
    r"avg_done_rate=(?P<done>[-+0-9.eE]+)"
)


DEFAULT_CONDITIONS = [
    ("train_like", []),
    (
        "clean",
        [
            "task.env.randomization.obs_noise_e_scale=0.0",
            "task.env.randomization.obs_noise_t_scale=0.0",
            "task.env.forceScale=0.0",
            "task.env.randomForceProbScalar=0.0",
        ],
    ),
    (
        "light",
        [
            "task.env.randomization.obs_noise_e_scale=0.03",
            "task.env.randomization.obs_noise_t_scale=0.015",
            "task.env.forceScale=1.0",
            "task.env.randomForceProbScalar=0.2",
        ],
    ),
    (
        "hard",
        [
            "task.env.randomization.obs_noise_e_scale=0.05",
            "task.env.randomization.obs_noise_t_scale=0.025",
            "task.env.forceScale=1.5",
            "task.env.randomForceProbScalar=0.3",
        ],
    ),
]


def parse_condition(raw):
    parts = [part for part in raw.split("::")]
    name = parts[0].strip()
    if not name:
        raise ValueError(f"empty condition name in {raw!r}")
    return name, [part.strip() for part in parts[1:] if part.strip()]


def expand_checkpoints(patterns):
    out = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and os.path.exists(pattern):
            matches = [pattern]
        for match in matches:
            path = os.path.abspath(match)
            if path not in seen:
                seen.add(path)
                out.append(path)
    return out


def safe_label(path):
    p = Path(path)
    parent = p.parent.name
    stem = p.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{parent}_{stem}")


def build_command(args, checkpoint, condition_overrides):
    runner = args.runner.split() if args.runner else [sys.executable]
    checkpoint_arg = checkpoint
    if args.runner and os.path.isabs(checkpoint):
        rel = os.path.relpath(checkpoint, os.getcwd())
        if not rel.startswith(".."):
            checkpoint_arg = rel
    cmd = [
        *runner,
        "python" if args.runner else "train.py",
    ]
    if args.runner:
        cmd.append("train.py")
    cmd.extend(
        [
            f"task={args.task}",
            f"train.algo={args.algo}",
            "test=True",
            "headless=True",
            "wandb_activate=False",
            f"seed={args.seed}",
            f"checkpoint={checkpoint_arg}",
            f"+test_num_steps={args.steps}",
            f"task.env.numEnvs={args.num_envs}",
            f"train.ppo.output_name={args.output_name}",
        ]
    )
    if args.graphics_device_id is not None:
        cmd.append(f"graphics_device_id={args.graphics_device_id}")
    cmd.extend(args.extra_override)
    cmd.extend(condition_overrides)
    return cmd


def run_one(cmd, log_path, dry_run=False):
    if dry_run:
        text = "DRY_RUN " + " ".join(cmd) + "\n"
        Path(log_path).write_text(text)
        return 0, text
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_f.write(proc.stdout)
    return proc.returncode, proc.stdout


def parse_eval_summary(text):
    matches = list(EVAL_RE.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return {
        "steps": int(m.group("steps")),
        "avg_reward": float(m.group("reward")),
        "avg_done_rate": float(m.group("done")),
    }


def deploy_name_for(path, requested):
    suffix = Path(path).suffix
    if requested:
        stem = requested
    elif suffix == ".pth":
        stem = "best_deploy"
    else:
        stem = "model_best_deploy"
    return str(Path(path).with_name(f"{stem}{suffix}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--checkpoints", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--output-name", default="eval_select_tmp")
    parser.add_argument("--runner", default="./docker-run-isaacgym.sh")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--done-penalty", type=float, default=2000.0)
    parser.add_argument("--deploy-name", default="")
    parser.add_argument("--graphics-device-id", type=int, default=None)
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="Format: name::override1::override2. Defaults to train_like/clean/light/hard.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoints = expand_checkpoints(args.checkpoints)
    if not checkpoints:
        raise SystemExit("No checkpoints matched.")
    conditions = [parse_condition(c) for c in args.condition] if args.condition else DEFAULT_CONDITIONS

    out_dir = Path(args.out_dir)
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for checkpoint in checkpoints:
        ckpt_label = safe_label(checkpoint)
        for cond_name, cond_overrides in conditions:
            log_path = log_dir / f"{ckpt_label}__{cond_name}.log"
            cmd = build_command(args, checkpoint, cond_overrides)
            start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            status, text = run_one(cmd, log_path, dry_run=args.dry_run)
            parsed = parse_eval_summary(text)
            avg_reward = parsed["avg_reward"] if parsed else float("nan")
            avg_done_rate = parsed["avg_done_rate"] if parsed else float("nan")
            score = avg_reward - args.done_penalty * avg_done_rate
            rows.append(
                {
                    "time": start,
                    "checkpoint": checkpoint,
                    "condition": cond_name,
                    "status": status,
                    "steps": parsed["steps"] if parsed else "",
                    "avg_reward": avg_reward,
                    "avg_done_rate": avg_done_rate,
                    "score": score,
                    "log": str(log_path),
                    "command": " ".join(cmd),
                }
            )
            print(
                f"{ckpt_label} {cond_name}: status={status} "
                f"reward={avg_reward:.6f} done={avg_done_rate:.6f} score={score:.6f}"
            )

    summary_path = out_dir / "summary.tsv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    by_ckpt = {}
    for row in rows:
        by_ckpt.setdefault(row["checkpoint"], []).append(row)

    ranking = []
    for checkpoint, ckpt_rows in by_ckpt.items():
        valid = [r for r in ckpt_rows if r["status"] == 0 and r["score"] == r["score"]]
        if len(valid) != len(ckpt_rows):
            mean_score = -float("inf")
            mean_reward = -float("inf")
            mean_done = float("inf")
        else:
            mean_score = sum(float(r["score"]) for r in valid) / len(valid)
            mean_reward = sum(float(r["avg_reward"]) for r in valid) / len(valid)
            mean_done = sum(float(r["avg_done_rate"]) for r in valid) / len(valid)
        ranking.append((mean_score, mean_reward, -mean_done, checkpoint))
    ranking.sort(reverse=True)

    rank_path = out_dir / "ranking.tsv"
    with open(rank_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["rank", "checkpoint", "mean_score", "mean_reward", "mean_done_rate"])
        for idx, (mean_score, mean_reward, neg_mean_done, checkpoint) in enumerate(ranking, 1):
            writer.writerow([idx, checkpoint, mean_score, mean_reward, -neg_mean_done])

    finite_ranking = [item for item in ranking if math.isfinite(item[0])]
    if not finite_ranking:
        msg = "No checkpoint produced a valid EvalSummary for every condition."
        if args.dry_run:
            print(f"DRY_RUN {msg}")
            print(f"summary={summary_path}")
            print(f"ranking={rank_path}")
            return
        raise SystemExit(msg)

    best = finite_ranking[0][3]
    dst = deploy_name_for(best, args.deploy_name)
    if not args.dry_run:
        shutil.copy2(best, dst)
    print(f"EvalSelectDeployBest checkpoint={best} deploy_path={dst}")
    print(f"summary={summary_path}")
    print(f"ranking={rank_path}")


if __name__ == "__main__":
    main()
