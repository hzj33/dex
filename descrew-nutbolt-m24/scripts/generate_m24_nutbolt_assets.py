#!/usr/bin/env python3
"""Generate metric M24x3 nut/bolt meshes and a 100-point task cloud.

Dimensions are metres internally:
- nut: 36 mm across flats, 21.5 mm thick, M24x3 clearance bore
- bolt: 24 mm major diameter, 3 mm pitch, 160 mm shaft length
- bolt head: 36 mm across flats, 15 mm high

The detailed bolt thread is visual-only. The URDF uses a cylindrical shaft
collision shape for stable Isaac Gym contact simulation.
"""

import argparse
import math
import struct
from pathlib import Path

import numpy as np


NUT_ACROSS_FLATS = 0.036
NUT_THICKNESS = 0.0215
NUT_BORE_RADIUS = 0.01225
BOLT_MAJOR_RADIUS = 0.012
BOLT_MINOR_RADIUS = 0.0105
BOLT_PITCH = 0.003
BOLT_LENGTH = 0.160
HEAD_ACROSS_FLATS = 0.036
HEAD_HEIGHT = 0.015


def _normal(a, b, c):
    normal = np.cross(b - a, c - a)
    length = np.linalg.norm(normal)
    return normal / length if length > 1e-12 else np.zeros(3, dtype=np.float64)


def _quad(triangles, a, b, c, d):
    triangles.append((np.asarray(a), np.asarray(b), np.asarray(c)))
    triangles.append((np.asarray(a), np.asarray(c), np.asarray(d)))


def _write_binary_stl(path, name, triangles):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = name.encode("ascii", errors="replace")[:80].ljust(80, b" ")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            normal = _normal(a, b, c)
            values = [*normal, *a, *b, *c]
            stream.write(struct.pack("<12fH", *values, 0))


def _hex_perimeter(across_flats, samples_per_face=8):
    radius = (across_flats / 2.0) / math.cos(math.pi / 6.0)
    vertices = [
        np.array(
            [radius * math.cos(math.pi / 6 + i * math.pi / 3),
             radius * math.sin(math.pi / 6 + i * math.pi / 3)],
            dtype=np.float64,
        )
        for i in range(6)
    ]
    points = []
    for face in range(6):
        start = vertices[face]
        end = vertices[(face + 1) % 6]
        for sample in range(samples_per_face):
            alpha = sample / samples_per_face
            points.append(start * (1.0 - alpha) + end * alpha)
    return np.asarray(points)


def make_nut_mesh():
    base_outer = _hex_perimeter(NUT_ACROSS_FLATS)
    count = len(base_outer)
    z_levels = np.asarray(
        [-NUT_THICKNESS / 2, -NUT_THICKNESS / 2 + 0.002,
         NUT_THICKNESS / 2 - 0.002, NUT_THICKNESS / 2]
    )
    outer_scales = np.asarray([34.0 / 36.0, 1.0, 1.0, 34.0 / 36.0])
    inner_radii = np.asarray(
        [NUT_BORE_RADIUS + 0.0006, NUT_BORE_RADIUS,
         NUT_BORE_RADIUS, NUT_BORE_RADIUS + 0.0006]
    )
    angles = np.arctan2(base_outer[:, 1], base_outer[:, 0])
    outer = [base_outer * scale for scale in outer_scales]
    inner = [
        np.column_stack((np.cos(angles) * radius, np.sin(angles) * radius))
        for radius in inner_radii
    ]
    triangles = []
    for layer in range(len(z_levels) - 1):
        for i in range(count):
            j = (i + 1) % count
            _quad(
                triangles,
                [*outer[layer][i], z_levels[layer]],
                [*outer[layer][j], z_levels[layer]],
                [*outer[layer + 1][j], z_levels[layer + 1]],
                [*outer[layer + 1][i], z_levels[layer + 1]],
            )
            _quad(
                triangles,
                [*inner[layer][j], z_levels[layer]],
                [*inner[layer][i], z_levels[layer]],
                [*inner[layer + 1][i], z_levels[layer + 1]],
                [*inner[layer + 1][j], z_levels[layer + 1]],
            )
    for i in range(count):
        j = (i + 1) % count
        _quad(
            triangles,
            [*inner[0][i], z_levels[0]],
            [*inner[0][j], z_levels[0]],
            [*outer[0][j], z_levels[0]],
            [*outer[0][i], z_levels[0]],
        )
        _quad(
            triangles,
            [*outer[-1][i], z_levels[-1]],
            [*outer[-1][j], z_levels[-1]],
            [*inner[-1][j], z_levels[-1]],
            [*inner[-1][i], z_levels[-1]],
        )
    return triangles


def make_hex_head_mesh():
    ring = _hex_perimeter(HEAD_ACROSS_FLATS, samples_per_face=1)
    bottom = -HEAD_HEIGHT
    top = 0.0
    center_bottom = np.array([0.0, 0.0, bottom])
    center_top = np.array([0.0, 0.0, top])
    triangles = []
    for i in range(6):
        j = (i + 1) % 6
        _quad(
            triangles,
            [*ring[i], bottom], [*ring[j], bottom],
            [*ring[j], top], [*ring[i], top],
        )
        triangles.append((center_bottom, np.array([*ring[j], bottom]), np.array([*ring[i], bottom])))
        triangles.append((center_top, np.array([*ring[i], top]), np.array([*ring[j], top])))
    return triangles


def make_thread_mesh():
    angular_samples = 48
    axial_samples = int(math.ceil(BOLT_LENGTH / BOLT_PITCH * 8)) + 1
    angles = np.linspace(0.0, 2.0 * math.pi, angular_samples, endpoint=False)
    heights = np.linspace(0.0, BOLT_LENGTH, axial_samples)
    grid = []
    for z in heights:
        phase = angles - 2.0 * math.pi * z / BOLT_PITCH
        radius = (BOLT_MAJOR_RADIUS + BOLT_MINOR_RADIUS) / 2.0
        radius += (BOLT_MAJOR_RADIUS - BOLT_MINOR_RADIUS) / 2.0 * np.cos(phase)
        grid.append(np.column_stack((radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, z))))
    triangles = []
    for layer in range(axial_samples - 1):
        for i in range(angular_samples):
            j = (i + 1) % angular_samples
            _quad(triangles, grid[layer][i], grid[layer][j], grid[layer + 1][j], grid[layer + 1][i])
    return triangles


def _sample_mesh_surface(triangles, count, rng):
    tri = np.asarray(triangles, dtype=np.float64)
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    chosen = rng.choice(len(tri), size=count, p=areas / areas.sum())
    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    selected = tri[chosen]
    return selected[:, 0] + u[:, None] * (selected[:, 1] - selected[:, 0]) + v[:, None] * (selected[:, 2] - selected[:, 0])


def make_point_cloud(nut_triangles, nut_joint_z=0.125):
    rng = np.random.default_rng(42)
    bolt_count = 30
    nut_count = 70
    angles = rng.uniform(0.0, 2.0 * math.pi, bolt_count)
    z = rng.uniform(0.0, BOLT_LENGTH, bolt_count)
    bolt = np.column_stack((BOLT_MAJOR_RADIUS * np.cos(angles), BOLT_MAJOR_RADIUS * np.sin(angles), z))
    nut = _sample_mesh_surface(nut_triangles, nut_count, rng)
    nut[:, 2] += nut_joint_z
    return np.vstack((bolt, nut)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", default="assets")
    args = parser.parse_args()
    asset_root = Path(args.asset_root)
    mesh_dir = asset_root / "meshes"
    object_dir = asset_root / "screw" / "m24hex"
    nut = make_nut_mesh()
    _write_binary_stl(mesh_dir / "m24x3_hex_nut_36x21p5.stl", "M24x3 hex nut", nut)
    _write_binary_stl(mesh_dir / "m24_hex_bolt_head_36x15.stl", "M24 bolt head", make_hex_head_mesh())
    _write_binary_stl(mesh_dir / "m24x3_bolt_thread_160.stl", "M24x3 bolt thread", make_thread_mesh())
    object_dir.mkdir(parents=True, exist_ok=True)
    np.save(object_dir / "0000_m24x3_160.npy", make_point_cloud(nut))
    print(f"Generated M24 assets under {asset_root}")


if __name__ == "__main__":
    main()
