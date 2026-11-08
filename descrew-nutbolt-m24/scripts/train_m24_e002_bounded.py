#!/usr/bin/env python3
"""Existing PPO entrypoint with completed-update snapshots for bounded E002 runs.

SIGTERM requests an orderly stop after the current update. Every completed
update already has a snapshot, even if the timeout's kill-after is reached.
The PPO algorithm, optimizer and reward are not modified by this wrapper.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as entry  # Isaac Gym is imported before torch by the existing entrypoint.

stop_requested = False
original_epoch = entry.PPO.train_epoch


def request_stop(signum, frame):
    global stop_requested
    stop_requested = True


def completed_epoch(self):
    result = original_epoch(self)
    if not self.env.e002_motion_enabled:
        raise ValueError("bounded E002 wrapper requires the E002 task")
    root = Path(self.nn_dir)
    temporary = root / "last_completed.tmp"
    self.save(str(temporary))
    os.replace(str(temporary) + ".pth", root / "last.pth")
    metadata = dict(epoch=self.epoch_num, agent_steps=self.agent_steps,
        completed_at=datetime.now(timezone.utc).isoformat(), stopped_by_timeout=stop_requested,
        checkpoint="last.pth", snapshot_boundary="after completed PPO update")
    (root / "last.training_state.tmp").write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(root / "last.training_state.tmp", root / "last.training_state.json")
    interval = float(self.ppo_config.get('snapshot_interval_seconds', 0.0))
    if interval > 0 and time.monotonic() - getattr(self, '_last_periodic_snapshot', 0.0) >= interval:
        snapshots = root / 'periodic'
        snapshots.mkdir(exist_ok=True)
        self.save(str(snapshots / ('epoch_%06d' % self.epoch_num)))
        self._last_periodic_snapshot = time.monotonic()
    self.writer.flush()
    if stop_requested:
        self.writer.close()
        print("[bounded E002] completed update saved; stopping on SIGTERM", flush=True)
        raise SystemExit(0)
    return result


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_stop)
    entry.PPO.train_epoch = completed_epoch
    # Imported Hydra entrypoints resolve relative paths as Python modules.
    # Supply the repo config directory explicitly while keeping main unchanged.
    main = entry.hydra.main(config_name="config",
        config_path=str(Path(__file__).resolve().parents[1] / "configs"))(
            entry.main.__wrapped__)
    main()
