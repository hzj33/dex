import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pxdex.dh13 import ControlMode
from pxdex.dh13 import DexH13Control

# =========================
# Model & Device Config
# =========================

# JIT_POLICY_PATH = (
#     "/home/vision/dex-bulb/deploy/codrive(1)/codrive/bc_latentbc_deploy/model_best.pt"
# )
# JIT_POLICY_PATH = "/home/vision/dex-bulb/deploy/codrive(1)/codrive/consistency_latent_deploy/model_best.pt"
# JIT_POLICY_PATH = (
#     "/home/vision/dex-bulb/deploy/codrive(1)/codrive/dagger_deploy/model_best.pt"
# )
# JIT_POLICY_PATH = "/home/vision/dex-bulb/deploy/codrive(1)/codrive/diffusion_action_chunk_deploy/model_best.pt"
# JIT_POLICY_PATH = "/home/vision/dex-bulb/deploy/codrive(1)/codrive/diffusion_latent_deploy/model_best.pt"
# JIT_POLICY_PATH = (
#     "/home/vision/dex-bulb/deploy/codrive(1)/codrive/dotpg_deployv1/model_best.pt"
# )
# JIT_POLICY_PATH = (
#     "/home/vision/dex-bulb/deploy/codrive(1)/codrive/flow_matching_deploy/model_best.pt"
# )
JIT_POLICY_PATH = "/home/vision123/software/DOTPG/dex-bulb/sim2real/codrive/model_best.pt"


# JIT_POLICY_PATH = "/home/vision/dex-bulb/deploy/codrive_noise/padapt/model_best_codrive.pt"
# JIT_POLICY_PATH = "/home/vision/dex-bulb/deploy/codrive/padapt/model_best_codrive.pt"
# JIT_POLICY_PATH = "/home/vision/dex-bulb/sim2real/codrive/codrive_noise/codrive_noise_dotpg_deploy/model_best.pt"

DEVICE = torch.device("cpu")

HANDY_PORT = "/dev/ttyUSB0"
CAMERA_PORT = "auto"

ACTION_SCALE = 0.065
LOCKED_ZERO_JOINT_INDICES = (4, 5, 6, 7, 8, 9, 10, 11)
# LOCKED_ZERO_JOINT_INDICES = (3,7,11)

Q_HOLD_EPS = 0.05
POLICY_DT = 1.0 / 20.0  # 20 Hz

try:
    jit_model = torch.jit.load(JIT_POLICY_PATH, map_location=DEVICE)
except RuntimeError as exc:
    if "constants.pkl" in str(exc):
        raise RuntimeError(
            f"{JIT_POLICY_PATH} is not a TorchScript/JIT .pt file. "
            "This deploy script needs a model saved by torch.jit.save(). "
            "A plain state_dict checkpoint saved by torch.save() cannot be used here."
        ) from exc
    raise
jit_model.eval()
print(f"Successfully loaded JIT policy from {JIT_POLICY_PATH}")


# =========================
# Observation / Proprio Config
# =========================

OBS_DIM = 96
PROPRIO_HIST_LEN = 30
NUM_DOF = 16
PROPRIO_DIM = NUM_DOF * 2

dexh13_dof_lower_limits = np.array(
    [-0.34, 0.0, 0.0, 0.0, -0.001, 0.0, 0.0, 0.0, -0.001, 0.0, 0.0, 0.0, -0.34, 0.0, 0.0, 0.0], dtype=np.float32
)

dexh13_dof_upper_limits = np.array(
    [0.34, 1.56, 1.56, 1.56, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.34, 1.56, 1.56, 1.56],
    dtype=np.float32,
)

base = torch.zeros((1, PROPRIO_HIST_LEN, PROPRIO_DIM), device=DEVICE, dtype=torch.float32)
proprio_hist_buffer = base.clone()

# =========================
# Normalization stats from model
# =========================

obs_mean = jit_model.running_mean.unsqueeze(0).to(device=DEVICE, dtype=torch.float32)
obs_var = jit_model.running_var.unsqueeze(0).to(device=DEVICE, dtype=torch.float32)
sa_mean = jit_model.sa_mean.unsqueeze(0).to(device=DEVICE, dtype=torch.float32)
sa_var = jit_model.sa_var.unsqueeze(0).to(device=DEVICE, dtype=torch.float32)


def update_proprio_hist(q, target_q):
    global proprio_hist_buffer

    q = np.asarray(q, dtype=np.float32)
    target_q = np.asarray(target_q, dtype=np.float32)

    current_dof = torch.from_numpy(q).float().to(DEVICE)
    target_dof = torch.from_numpy(target_q).float().to(DEVICE)
    current_proprio = torch.cat([current_dof, target_dof], dim=0).unsqueeze(0).unsqueeze(0)
    prev_obs_buf = proprio_hist_buffer[:, 1:].clone()
    proprio_hist_buffer.copy_(torch.cat([prev_obs_buf, current_proprio], dim=1))


def reset_proprio_hist(q, target_q):
    global proprio_hist_buffer

    q = np.asarray(q, dtype=np.float32)
    target_q = np.asarray(target_q, dtype=np.float32)
    current_proprio = torch.from_numpy(np.concatenate([q, target_q]).astype(np.float32)).to(DEVICE)
    proprio_hist_buffer.copy_(current_proprio.view(1, 1, PROPRIO_DIM).repeat(1, PROPRIO_HIST_LEN, 1))


def make_obs_from_proprio_hist():
    return proprio_hist_buffer[:, -3:, :].reshape(1, OBS_DIM).clone()


def clamp_locked_joints(q):
    q = np.asarray(q, dtype=np.float32).copy()
    q[list(LOCKED_ZERO_JOINT_INDICES)] = 0.0
    return q


class DexH13DeployControl:
    def __init__(self):
        self.control = DexH13Control()
        self._last_q = np.zeros(NUM_DOF, dtype=np.float32)

    @staticmethod
    def _camera_port_candidates(camera_port):
        if camera_port not in ("auto", "", None, "/dev/video*"):
            return [camera_port]

        def video_index(path):
            match = re.search(r"/dev/video(\d+)$", str(path))
            return int(match.group(1)) if match else 1_000_000

        ports = sorted(Path("/dev").glob("video*"), key=video_index)
        candidates = [
            str(port) for port in ports if re.match(r"^/dev/video\d+$", str(port))
        ]
        return candidates or ["/dev/video0"]

    def open_device(self):
        print("//================================")
        print("// Open DexH13 device")
        print("//================================")

        handy_type = None
        errors = []
        for camera_port in self._camera_port_candidates(CAMERA_PORT):
            print(f"=@= trying camera port: {camera_port}")
            self.control = DexH13Control()
            try:
                handy_type = self.control.activeHandy(HANDY_PORT, camera_port)
                print(
                    f"=@= activeHandy done, handy_type={handy_type}, "
                    f"camera_port={camera_port}"
                )
                break
            except RuntimeError as exc:
                errors.append((camera_port, exc))
                print(f"=@= activeHandy failed on {camera_port}: {exc}")
        else:
            tried = ", ".join(port for port, _ in errors) or str(CAMERA_PORT)
            raise RuntimeError(f"PaXini DexH13 SDK 提示：所有摄像头端口都打开失败: {tried}") from (
                errors[-1][1] if errors else None
            )

        if not self.control.isConnectHandy():
            return False

        print(f"=@= initMotorPosition: {self.control.initMotorPosition()}")
        print(
            f"=@= set POSITION_CONTROL_MODE: " f"{self.control.setMotorControlMode(ControlMode.POSITION_CONTROL_MODE)}"
        )
        print(f"=@= enableMotor: {self.control.enableMotor()}")
        print(f"=@= isMotorEnabled: {self.control.isMotorEnabled()}")
        return self.control.isMotorEnabled()

    def send_joint_target(self, q_target):
        return self.control.setJointPositionsRadian(np.asarray(q_target, dtype=np.float32).tolist())

    def read_joint_pos(self):
        q = np.asarray(self.control.getJointPositionsRadian(), dtype=np.float32).reshape(-1)
        if q.size == 0:
            # print("关节读数为空")
            return self._last_q.copy()
        q = np.round(q[:NUM_DOF], 4)
        # hold_mask = np.abs(q - self._last_q) < Q_HOLD_EPS
        # q[hold_mask] = self._last_q[hold_mask]
        self._last_q = q.copy()
        return q
    
    def read_finger_tactile(self):
        finger_tactile_list = self.control.getFingerTactile()
        for finger_tactile in finger_tactile_list:
            print(f"x:{finger_tactile.x}, y:{finger_tactile.y}, z:{finger_tactile.z}")
        print("================================")
        

    def close(self):
        print(f"=@= disableMotor: {self.control.disableMotor()}")
        print(f"=@= disconnectHandy: {self.control.disconnectHandy()}")


def main():
    controller = DexH13DeployControl()
    if not controller.open_device():
        sys.exit(1)

    try:
        # prev_target_policy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        prev_target_policy = [0.34, 0.92, 0.28, 1.13, 0, 0, 0, 0, 0, 0, 0, 0, -0.34, 1.56, 0.22, 0.41]

        controller.send_joint_target(prev_target_policy)
        time.sleep(1.0)

        prev_target = np.asarray(prev_target_policy, dtype=np.float32)
        target_q = prev_target.copy()
        q_init = clamp_locked_joints(controller.read_joint_pos())
        reset_proprio_hist(q_init, prev_target)
        last_policy_time = time.perf_counter() - POLICY_DT  # 让第一步立即触发

        while True:
            q_dexh13 = controller.read_joint_pos()
            q_policy = clamp_locked_joints(q_dexh13)
            controller.read_finger_tactile()

            now = time.perf_counter()
            if (
                now - last_policy_time >= POLICY_DT
            ):  # 确保policy的推理频率为20Hz(paxini的读写操作总共需要30ms，若按原项目的写法将导致推理频率变为3Hz)
                last_policy_time = now

                update_proprio_hist(q_policy, prev_target)
                obs = make_obs_from_proprio_hist()
                with torch.no_grad():
                    obs = torch.clamp(obs, -5.0, 5.0)

                    norm_obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-5)
                    norm_obs = torch.clamp(norm_obs, -5.0, 5.0)

                    norm_proprio_hist = (proprio_hist_buffer - sa_mean) / torch.sqrt(sa_var + 1e-5)
                    norm_proprio_hist = torch.clamp(norm_proprio_hist, -5.0, 5.0)
                    point_cloud_info = torch.zeros((1, 100, 3), device=DEVICE)

                    input_dict = {
                        "obs": norm_obs,
                        "proprio_hist": norm_proprio_hist,
                        "point_cloud_info": point_cloud_info,
                    }

                    mu, _, _ = jit_model(input_dict)
                    mu = torch.clamp(mu, -1.0, 1.0)

                    jit_action = mu.detach().cpu().numpy().flatten()
                target_q = jit_action * ACTION_SCALE + prev_target
                target_q = np.clip(target_q, dexh13_dof_lower_limits, dexh13_dof_upper_limits)
                target_q = clamp_locked_joints(target_q)
                prev_target = target_q.copy()

            controller.send_joint_target(target_q)

    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt, exiting...")
        controller.send_joint_target([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        time.sleep(1)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
