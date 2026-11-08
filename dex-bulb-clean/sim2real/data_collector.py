from __future__ import annotations

import datetime as _dt
import pickle
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class Sim2RealDataCollector:
    """Save synchronized sim2real deployment frames as individual pickle files."""

    def __init__(
        self,
        root_dir: str | Path,
        run_name: Optional[str] = None,
        async_write: bool = True,
        max_queue_size: int = 2048,
    ):
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(root_dir).expanduser()
        if run_name:
            self.save_dir = self.save_dir / run_name
        else:
            self.save_dir = self.save_dir / timestamp
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._last_time: Optional[float] = None
        self._frequencies: list[float] = []
        self._async_write = bool(async_write)
        self._queue: queue.Queue[tuple[str, Dict[str, Any]] | None] = queue.Queue(
            maxsize=int(max_queue_size)
        )
        self._writer_error: Optional[BaseException] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._dropped_frames = 0
        if self._async_write:
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()
        print(f"[DataCollector] Saving frames to: {self.save_dir}")
        print(f"[DataCollector] async_write={self._async_write}")

    def _write_pickle(self, filename: str, data: Dict[str, Any]) -> None:
        with open(self.save_dir / filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                filename, data = item
                self._write_pickle(filename, data)
            except BaseException as exc:
                self._writer_error = exc
            finally:
                self._queue.task_done()

    def save_frame(
        self,
        *,
        timestamp: _dt.datetime,
        policy_step: int,
        hand_side: str,
        q_hand: np.ndarray,
        target_q: np.ndarray,
        action: np.ndarray,
        tactile: np.ndarray,
        obs: np.ndarray,
        proprio_hist: np.ndarray,
        q_policy: np.ndarray,
        arm_state: Dict[str, Any],
        wrist_state: Dict[str, Any],
        tactile_detail: Optional[np.ndarray] = None,
        policy_running: bool = True,
        sim_action_scale: float = 0.05,
        sim_control_dt: float = 0.05,
    ) -> None:
        now = time.perf_counter()
        dt = 0.0 if self._last_time is None else now - self._last_time
        self._last_time = now
        if dt > 0:
            self._frequencies.append(1.0 / dt)

        data: Dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "policy_step": int(policy_step),
            "dt": float(dt),
            "hand_side": hand_side,
            "paxini_pos": np.asarray(q_hand, dtype=np.float32).reshape(16).copy(),
            "paxini_act": np.asarray(target_q, dtype=np.float32).reshape(16).copy(),
            "paxini_tactile": np.asarray(tactile, dtype=np.int32).reshape(33).copy(),
            "paxini_tactile_detail": np.asarray(
                np.zeros(0, dtype=np.int32) if tactile_detail is None else tactile_detail,
                dtype=np.int32,
            )
            .reshape(-1)
            .copy(),
            "paxini_rl_flag": bool(policy_running),
            "control": np.asarray(action, dtype=np.float32).reshape(16).copy(),
            "action": np.asarray(action, dtype=np.float32).reshape(16).copy(),
            "actions": np.asarray(action, dtype=np.float32).reshape(16).copy(),
            "obs": np.asarray(obs, dtype=np.float32).reshape(-1).copy(),
            "proprio_hist": np.asarray(proprio_hist, dtype=np.float32).reshape(30, 32).copy(),
            "point_cloud_info": np.zeros((100, 3), dtype=np.float32),
            "q_policy": np.asarray(q_policy, dtype=np.float32).reshape(16).copy(),
            "hand_dof_pos": np.asarray(q_policy, dtype=np.float32).reshape(16).copy(),
            "hand_dof_target": np.asarray(target_q, dtype=np.float32).reshape(16).copy(),
            "hand_joint_pos": np.asarray(q_policy, dtype=np.float32).reshape(16).copy(),
            "sim_action_scale": float(sim_action_scale),
            "sim_control_dt": float(sim_control_dt),
            "activated": bool(policy_running),
            "switch": False,
        }
        data.update(arm_state)
        data.update(wrist_state)
        for key, value in list(data.items()):
            if isinstance(value, np.ndarray):
                data[key] = value.copy()

        filename = timestamp.strftime("%Y%m%d_%H%M%S_%f") + ".pkl"
        if not self._async_write:
            self._write_pickle(filename, data)
            return

        if self._writer_error is not None:
            raise RuntimeError("Data writer thread failed.") from self._writer_error
        try:
            self._queue.put_nowait((filename, data))
        except queue.Full:
            self._dropped_frames += 1
            if self._dropped_frames in (1, 10, 100):
                print(
                    "[DataCollector] Warning: writer queue full; "
                    f"dropped_frames={self._dropped_frames}"
                )

    def save_statistics(self) -> None:
        self.close()
        if not self._frequencies:
            return
        freq = np.asarray(self._frequencies, dtype=np.float32)
        stats_file = self.save_dir / "freq.txt"
        with open(stats_file, "w") as f:
            f.write(f"Average FPS: {float(np.mean(freq)):.2f}\n")
            f.write(f"Max FPS: {float(np.max(freq)):.2f}\n")
            f.write(f"Min FPS: {float(np.min(freq)):.2f}\n")
            f.write(f"Std FPS: {float(np.std(freq)):.2f}\n")
            f.write(f"Dropped frames: {self._dropped_frames}\n")
        print(f"[DataCollector] Statistics saved to: {stats_file}")

    def close(self) -> None:
        if not self._async_write or self._writer_thread is None:
            return
        self._queue.join()
        self._queue.put(None)
        self._writer_thread.join(timeout=5.0)
        self._writer_thread = None
        if self._writer_error is not None:
            raise RuntimeError("Data writer thread failed.") from self._writer_error
