"""URDF link-frame kinematics for offline diagnostics, not measured clearances."""
import xml.etree.ElementTree as ET
import numpy as np


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross @ cross)


class HandKinematics:
    def __init__(self, urdf):
        self.joints = []
        for joint in ET.parse(urdf).getroot().findall('joint'):
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            r, p, y = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
            transform = np.eye(4)
            transform[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
            transform[:3, 3] = xyz
            axis = joint.find('axis')
            self.joints.append((joint.get('name'), joint.get('type'),
                joint.find('parent').get('link'), joint.find('child').get('link'),
                transform, np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else None))
        self.order = ['right_%s_joint_%d' % (finger, j)
                      for finger in ('index', 'middle', 'ring', 'thumb') for j in range(4)]
        assert set(self.order) == {j[0] for j in self.joints if j[1] == 'revolute'}

    def frames(self, q):
        angles = dict(zip(self.order, q))
        result = {'right_palm_link': np.eye(4)}
        pending = list(self.joints)
        while pending:
            remaining = []
            for name, kind, parent, child, origin, axis in pending:
                if parent not in result:
                    remaining.append((name, kind, parent, child, origin, axis))
                    continue
                motion = np.eye(4)
                if kind == 'revolute':
                    motion[:3, :3] = rotation(axis, angles[name])
                else:
                    assert kind == 'fixed'
                result[child] = result[parent] @ origin @ motion
            assert len(remaining) < len(pending), 'Unresolved hand URDF tree'
            pending = remaining
        return result

    def separation(self, q):
        frames = self.frames(q)
        return {name: float(np.linalg.norm(frames[index][:3, 3]-frames[thumb][:3, 3]))
                for name, index, thumb in (
                    ('tip_link_origins', 'right_index_link_3', 'right_thumb_link_3'),
                    ('distal_tactile_origins', 'right_index_tactile_link_2', 'right_thumb_tactile_link_1'))}
