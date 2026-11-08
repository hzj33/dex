from dexscrew.tasks.xhand_hora import XHandHora


class RBHandHora(XHandHora):
    """RBHand 版本的 Hora 任务。

    目前实现仍复用 `XHandHora`，但保留独立的 class/file 可以避免命名歧义，
    也方便后续加入 RBHand 专属逻辑而不影响 XHand 相关任务。
    """
