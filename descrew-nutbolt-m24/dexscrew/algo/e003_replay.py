"""Local bounded replay adaptation: retain initial expert demonstrations."""
import torch


def sample_with_expert_anchor(replay, size, anchor, anchor_size, batch, fraction):
    assert 0 <= fraction <= 1 and size > 0 and batch > 0
    device = replay['action'].device
    ids = torch.randint(size, (batch,), device=device)
    sample = {key: value[ids] for key, value in replay.items()}
    count = min(batch, int(batch * fraction)) if anchor_size else 0
    if count:
        anchor_ids = torch.randint(anchor_size, (count,), device=device)
        for key in sample:
            sample[key][:count] = anchor[key][anchor_ids]
    return sample, count
