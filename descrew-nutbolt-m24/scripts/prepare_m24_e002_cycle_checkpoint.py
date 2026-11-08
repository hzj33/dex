#!/usr/bin/env python3
"""Initialize newly activated binary command statistics; keep model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    assert not args.destination.exists()
    weights = torch.load(args.source, map_location='cpu')
    stats = weights['running_mean_std']
    assert stats['running_mean'].shape == stats['running_var'].shape == (101,)
    before = {k: stats[k][96:100].tolist() for k in ('running_mean','running_var')}
    stats['running_mean'][96:100] = torch.tensor([1.,0.,0.,0.], dtype=stats['running_mean'].dtype)
    stats['running_var'][96:100] = 1.
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, args.destination)
    args.destination.with_suffix('.json').write_text(json.dumps(dict(
        source=str(args.source), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        changed_indices=[96,97,98,99], before=before, model_weights_changed=False,
        reason='Previously constant command bits had variance approximately1e-7; unseen phase bits would normalize to thousands.',
        destination_sha256=hashlib.sha256(args.destination.read_bytes()).hexdigest()), indent=2)+'\n')
