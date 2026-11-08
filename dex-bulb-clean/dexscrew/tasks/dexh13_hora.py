from dexscrew.tasks.xhand_hora import XHandHora


class Dexh13Hora(XHandHora):
    """Dexh13 版本的 Hora 任务。

    目前实现复用 `XHandHora` 的完整环境逻辑，仅通过 task YAML 指定：
    - 手模型 URDF 路径（env.asset.handAsset）
    - fingertipBodies / handInitPose / handRootPos 等参数

    保留独立的 class/file 便于后续加入 Dexh13 专属逻辑而不影响其他手模型。
    """

