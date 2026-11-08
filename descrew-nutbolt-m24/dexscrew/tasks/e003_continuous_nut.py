"""Opt-in continuous M24 rotation, without PhysX wrapped-limit rebound.

IsaacGym imports a revolute joint whose raw position wraps at +/-2pi.
A nonnegative lower limit then projects the wrapped position back to zero.
Use the same asset with a continuous nut joint and unwrap ONLY +/-2pi
boundary crossings (4pi raw discontinuities). Do not write unwrapped angles
to the physics state or loosen cycle/contact/jump gates.
"""
import hashlib
import math
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import torch


def instrument_continuous_nut(base):
    class ContinuousNut(base):
        def _create_object_asset(self):
            # Resolve original mesh paths before relocating the temporary URDF.
            assert self.config['env']['object']['type'] == 'screw_m24hex'
            assert len(self.object_type_list) == 1
            key = self.object_type_list[0]
            source = Path(self.asset_files_dict[key]).resolve()
            root = ET.parse(source)
            joint = root.find("./joint[@name='nut_joint']")
            assert joint is not None and joint.attrib['type'] == 'revolute'
            joint.set('type', 'continuous')
            for mesh in root.findall('.//mesh'):
                mesh.set('filename', str((source.parent/mesh.attrib['filename']).resolve()))
            dest = Path('outputs/e003_continuous_asset')/str(os.getpid())/source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            root.write(dest, encoding='unicode', xml_declaration=True)
            shutil.copy2(source.with_suffix('.npy'), dest.with_suffix('.npy'))
            self.asset_files_dict[key] = str(dest)
            self.continuous_nut_receipt = dict(
                source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                generated=str(dest), generated_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                nut_joint='continuous', physical_base_fixed=True,
                position_source='raw physics angle + 4pi boundary offset',
                physics_state_modified_by_unwrap=False)
            super()._create_object_asset()

        def _create_envs(self, *args):
            super()._create_envs(*args)
            # Continuous URDF import changes unused drive stiffness/damping
            # defaults; restore the original passive joint's physical properties.
            for env in self.envs:
                actor = self.gym.find_actor_handle(env, 'object')
                assert self.gym.get_actor_rigid_body_count(env, actor) == 3
                p = self.gym.get_actor_dof_properties(env, actor)
                assert len(p) == 1 and not p['hasLimits'][0]
                p['stiffness'][:] = 0
                p['damping'][:] = .01
                p['friction'][:] = .2
                self.gym.set_actor_dof_properties(env, actor, p)

        def _refresh_gym(self):
            super()._refresh_gym()
            raw = self.nut_dof_pos.clone()
            if not hasattr(self, '_nut_raw_previous'):
                self._nut_raw_previous = raw.clone()
                self._nut_wrap_offset = torch.zeros_like(raw)
            delta = raw-self._nut_raw_previous
            period = 4*math.pi
            crossing = delta.abs() > 2*math.pi
            self._nut_wrap_offset -= torch.where(
                crossing, torch.round(delta/period)*period, torch.zeros_like(delta))
            self._nut_raw_previous.copy_(raw)
            self.nut_raw_angle = raw
            self.nut_dof_pos = raw+self._nut_wrap_offset

        def reset_idx(self, ids):
            if hasattr(self, '_nut_wrap_offset'):
                self._nut_wrap_offset[ids] = 0
                self._nut_raw_previous[ids] = 0
            super().reset_idx(ids)

    return ContinuousNut
