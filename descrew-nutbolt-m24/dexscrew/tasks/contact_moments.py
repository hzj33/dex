"""Vectorized nut-specific normal contact moments from Gym CPU contact records.

The original net-contact tensor is empirically normal-only in this task.
This corrects its application-point geometry; it does not infer friction.
"""

import numpy as np


def collect_env_contacts(gym, envs):
    """Own a copy of each validated env API result and make env IDs explicit.

    The installed Gym simulation-wide API returned invalid body/env indices
    in the two-env validation. Never use that API as a reward source.
    """
    batches = []
    for index, env in enumerate(envs):
        records = gym.get_env_rigid_contacts(env).copy()
        same_env = (records["env0"] == records["env1"]) & (records["env0"] >= 0)
        records = records[same_env]
        records["env0"] = index
        records["env1"] = index
        batches.append(records)
    return np.concatenate(batches)


def normal_contact_wrench(contacts, nut_quaternion, nut_handle, pair_handles, bodies_per_env):
    """Return force/moment on nut, each (num_envs, 2, 3), and contact counts.

    Gym's simulation API may expose simulation body indices whereas its env
    API exposes local body indices. Uniform M24 environments have the same
    body layout, so modulo maps either representation to the checked handles.
    Both env IDs must agree; ground and inter-environment contacts are excluded.
    """
    quaternions = np.asarray(nut_quaternion, dtype=float)
    count = len(quaternions)
    force = np.zeros((count, 2, 3))
    moment = np.zeros_like(force)
    contact_count = np.zeros((count, 2), dtype=np.int64)
    body0, body1 = np.asarray(contacts["body0"]), np.asarray(contacts["body1"])
    if not len(body0):
        return force, moment, contact_count
    b0, b1 = body0 % bodies_per_env, body1 % bodies_per_env
    env0, env1 = np.asarray(contacts["env0"]), np.asarray(contacts["env1"])
    valid = ((body0 >= 0) & (body1 >= 0) & (env0 == env1) &
             (env0 >= 0) & (env0 < count) & (np.asarray(contacts["lambda"]) > 0))
    for finger, handle in enumerate(pair_handles):
        # An integer preserves the historical distal-body metric; a group
        # measures an entire finger, preventing load transfer from hiding it.
        hit0 = (b0 == handle) if np.isscalar(handle) else np.isin(b0, handle)
        hit1 = (b1 == handle) if np.isscalar(handle) else np.isin(b1, handle)
        select = valid & (((b0 == nut_handle) & hit1) |
                          ((b1 == nut_handle) & hit0))
        ids = env0[select].astype(np.int64)
        if not len(ids):
            continue
        nut_is_body0 = b0[select] == nut_handle
        normal = np.stack([contacts["normal"][k][select] for k in ("x", "y", "z")], axis=-1)
        # Empirically validated in the actual task: +lambda*normal on body0.
        fnut = normal * np.asarray(contacts["lambda"])[select, None] * np.where(nut_is_body0[:, None], 1., -1.)
        local0 = np.stack([contacts["localPos0"][k][select] for k in ("x", "y", "z")], axis=-1)
        local1 = np.stack([contacts["localPos1"][k][select] for k in ("x", "y", "z")], axis=-1)
        arm = np.where(nut_is_body0[:, None], local0, local1)
        q = quaternions[ids].copy()
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        arm = arm + 2 * np.cross(q[:, :3], np.cross(q[:, :3], arm) + q[:, 3:4] * arm)
        np.add.at(force[:, finger], ids, fnut)
        np.add.at(moment[:, finger], ids, np.cross(arm, fnut))
        np.add.at(contact_count[:, finger], ids, 1)
    if not np.isfinite(force).all() or not np.isfinite(moment).all():
        raise ValueError("nonfinite contact wrench")
    return force, moment, contact_count
