# M24x3 Hex Nut And Bolt Asset

This folder contains the IsaacGym task assembly for the user-specified metric
nut and bolt:

- thread: M24x3
- bolt shaft length: 160 mm
- nut across flats: 36 mm
- nut thickness: 21.5 mm
- black-oxide visual finish
- nut initial center height: 125 mm above the bolt-head top

`0000_m24x3_160.urdf` preserves the task's expected three-link contract:

- `base`: fixed 36 mm hex bolt head
- `bolt`: fixed 160 mm M24 shaft
- `nut`: rotating hex nut connected by `nut_joint`

The bolt thread is a visual mesh. Its collision geometry is a 24 mm cylinder
for stable large-batch PhysX training. The nut uses a through-hole mesh for
both visual and collision geometry. As in the original NutBolt task,
`nut_joint` models rotation only; it does not translate the nut by 3 mm per
revolution.

Regenerate the STL meshes and 100-point task cloud from the repository root:

```bash
python scripts/generate_m24_nutbolt_assets.py
```
