"""
BC (Behavior Cloning) offline demonstration buffer.

定位
----
- 纯离线 BC 的固定专家演示数据集。
- 与 DAgger 的在线聚合 buffer 语义不同：BC buffer 只在训练开始前用 teacher 策略采集一次，
  训练期间不再追加 / 聚合 / 标签重采样（非 DAgger、非 BCO）。
- 字段与 DAggerBuffer 对齐，便于共用下游监督训练代码：
    obs, proprio_hist, teacher_action, teacher_extrin
    (optional) priv_info, point_cloud_info

存储约定
--------
- 默认存 CPU 以便放大数据集；dtype 可配 float16 / float32（默认 float32 以避免 1e-3
  的量化误差给 BC loss 加下限）
- 采样时通过 `_to_return` 统一转换到 self.device + return_dtype，供监督训练使用
"""

from typing import Optional, Tuple

import torch


class BCBuffer:
    """纯离线 BC 固定演示数据集（ring buffer，存 raw 输入 + teacher 标签）。"""

    def __init__(
        self,
        obs_dim: int,
        priv_info_dim: int,
        proprio_hist_shape: Tuple[int, int],
        point_cloud_shape: Optional[Tuple[int, int]],
        action_dim: int,
        teacher_extrin_dim: int,
        max_size: int = int(5e5),
        device: str = "cuda",
        storage_device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        return_dtype: torch.dtype = torch.float32,
        store_priv_info: bool = False,
        store_point_cloud: bool = False,
    ):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)
        self.storage_device = (
            torch.device(storage_device) if storage_device is not None else self.device
        )
        self.dtype = dtype
        self.return_dtype = return_dtype
        # 注意：store_priv_info / store_point_cloud 默认关闭。teacher_extrin 已作为
        # 标签直接入库，监督训练不再需要 priv / pc 来重建 teacher extrin。
        self.store_priv_info = bool(store_priv_info)
        self.store_point_cloud = bool(store_point_cloud) and point_cloud_shape is not None

        proprio_hist_shape = tuple(int(x) for x in proprio_hist_shape)
        self._proprio_hist_shape = proprio_hist_shape

        self.obs = torch.zeros(
            (self.max_size, int(obs_dim)), dtype=self.dtype, device=self.storage_device
        )
        self.proprio_hist = torch.zeros(
            (self.max_size, *proprio_hist_shape), dtype=self.dtype, device=self.storage_device
        )
        self.teacher_action = torch.zeros(
            (self.max_size, int(action_dim)), dtype=self.dtype, device=self.storage_device
        )
        self.teacher_extrin = torch.zeros(
            (self.max_size, int(teacher_extrin_dim)), dtype=self.dtype, device=self.storage_device
        )

        if self.store_priv_info:
            self.priv_info = torch.zeros(
                (self.max_size, int(priv_info_dim)),
                dtype=self.dtype,
                device=self.storage_device,
            )
        else:
            self.priv_info = None

        if self.store_point_cloud:
            pc_shape = tuple(int(x) for x in point_cloud_shape)
            self._point_cloud_shape = pc_shape
            self.point_cloud_info = torch.zeros(
                (self.max_size, *pc_shape),
                dtype=self.dtype,
                device=self.storage_device,
            )
        else:
            self._point_cloud_shape = None
            self.point_cloud_info = None

    def _cast(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.storage_device or x.dtype != self.dtype:
            x = x.to(device=self.storage_device, dtype=self.dtype)
        return x

    def add_batch(
        self,
        obs: torch.Tensor,
        proprio_hist: torch.Tensor,
        teacher_action: torch.Tensor,
        teacher_extrin: torch.Tensor,
        priv_info: Optional[torch.Tensor] = None,
        point_cloud_info: Optional[torch.Tensor] = None,
    ):
        obs = self._cast(obs.detach())
        proprio_hist = self._cast(proprio_hist.detach())
        teacher_action = self._cast(teacher_action.detach())
        teacher_extrin = self._cast(teacher_extrin.detach())
        if self.store_priv_info and priv_info is not None:
            priv_info = self._cast(priv_info.detach())
        if self.store_point_cloud and point_cloud_info is not None:
            point_cloud_info = self._cast(point_cloud_info.detach())

        batch_size = int(obs.shape[0])
        if batch_size == 0:
            return

        if self.ptr + batch_size <= self.max_size:
            self.obs[self.ptr:self.ptr + batch_size] = obs
            self.proprio_hist[self.ptr:self.ptr + batch_size] = proprio_hist
            self.teacher_action[self.ptr:self.ptr + batch_size] = teacher_action
            self.teacher_extrin[self.ptr:self.ptr + batch_size] = teacher_extrin
            if self.store_priv_info and priv_info is not None:
                self.priv_info[self.ptr:self.ptr + batch_size] = priv_info
            if self.store_point_cloud and point_cloud_info is not None:
                self.point_cloud_info[self.ptr:self.ptr + batch_size] = point_cloud_info
        else:
            first = self.max_size - self.ptr
            second = batch_size - first
            self.obs[self.ptr:] = obs[:first]
            self.proprio_hist[self.ptr:] = proprio_hist[:first]
            self.teacher_action[self.ptr:] = teacher_action[:first]
            self.teacher_extrin[self.ptr:] = teacher_extrin[:first]
            if self.store_priv_info and priv_info is not None:
                self.priv_info[self.ptr:] = priv_info[:first]
            if self.store_point_cloud and point_cloud_info is not None:
                self.point_cloud_info[self.ptr:] = point_cloud_info[:first]

            self.obs[:second] = obs[first:]
            self.proprio_hist[:second] = proprio_hist[first:]
            self.teacher_action[:second] = teacher_action[first:]
            self.teacher_extrin[:second] = teacher_extrin[first:]
            if self.store_priv_info and priv_info is not None:
                self.priv_info[:second] = priv_info[first:]
            if self.store_point_cloud and point_cloud_info is not None:
                self.point_cloud_info[:second] = point_cloud_info[first:]

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def _to_return(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device or x.dtype != self.return_dtype:
            x = x.to(device=self.device, dtype=self.return_dtype)
        return x

    def sample(self, batch_size: int):
        assert self.size > 0, "BCBuffer is empty"
        indices = torch.randint(0, self.size, (int(batch_size),), device=self.storage_device)

        out = {
            "obs": self._to_return(self.obs[indices]),
            "proprio_hist": self._to_return(self.proprio_hist[indices]),
            "teacher_action": self._to_return(self.teacher_action[indices]),
            "teacher_extrin": self._to_return(self.teacher_extrin[indices]),
        }
        if self.store_priv_info and self.priv_info is not None:
            out["priv_info"] = self._to_return(self.priv_info[indices])
        if self.store_point_cloud and self.point_cloud_info is not None:
            out["point_cloud_info"] = self._to_return(self.point_cloud_info[indices])
        return out

    def __len__(self):
        return self.size

    def clear(self):
        self.ptr = 0
        self.size = 0

    def save(self, path: str, meta: Optional[dict] = None):
        def materialize(x: torch.Tensor) -> torch.Tensor:
            # A sliced view still references the full preallocated storage.
            # Clone before torch.save so small demo buffers do not serialize
            # the entire capacity.
            return x[: self.size].detach().cpu().clone()

        payload = {
            "obs": materialize(self.obs),
            "proprio_hist": materialize(self.proprio_hist),
            "teacher_action": materialize(self.teacher_action),
            "teacher_extrin": materialize(self.teacher_extrin),
            "size": int(self.size),
            "max_size": int(self.max_size),
            "store_priv_info": bool(self.store_priv_info),
            "store_point_cloud": bool(self.store_point_cloud),
            "meta": dict(meta or {}),
        }
        if self.store_priv_info and self.priv_info is not None:
            payload["priv_info"] = materialize(self.priv_info)
        if self.store_point_cloud and self.point_cloud_info is not None:
            payload["point_cloud_info"] = materialize(self.point_cloud_info)
        torch.save(payload, path)

    def load(self, path: str) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        n = int(payload.get("size", 0))
        if n <= 0:
            return {}
        n = min(n, self.max_size)
        self.obs[:n] = payload["obs"][:n].to(device=self.storage_device, dtype=self.dtype)
        self.proprio_hist[:n] = payload["proprio_hist"][:n].to(device=self.storage_device, dtype=self.dtype)
        self.teacher_action[:n] = payload["teacher_action"][:n].to(device=self.storage_device, dtype=self.dtype)
        self.teacher_extrin[:n] = payload["teacher_extrin"][:n].to(device=self.storage_device, dtype=self.dtype)
        if self.store_priv_info and "priv_info" in payload and self.priv_info is not None:
            self.priv_info[:n] = payload["priv_info"][:n].to(device=self.storage_device, dtype=self.dtype)
        if self.store_point_cloud and "point_cloud_info" in payload and self.point_cloud_info is not None:
            self.point_cloud_info[:n] = payload["point_cloud_info"][:n].to(device=self.storage_device, dtype=self.dtype)
        self.size = n
        self.ptr = n % self.max_size
        self.meta = dict(payload.get("meta", {}) or {})
        return self.meta

    @property
    def meta(self) -> dict:
        return getattr(self, "_meta", {})

    @meta.setter
    def meta(self, value: dict):
        self._meta = dict(value or {})
