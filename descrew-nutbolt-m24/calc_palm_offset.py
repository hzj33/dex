# from isaacgym import gymapi
# import torch


# def get_palm_offset():
#     gym = gymapi.acquire_gym()
#     sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, gymapi.SimParams())

#     # Load Asset
#     asset_root = "."
#     asset_file = "assets/dexh13_right_description/urdf/dexh13_right_fix_path.urdf"
#     asset_options = gymapi.AssetOptions()
#     asset_options.fix_base_link = True
#     asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

#     # Create Env
#     env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

#     # Create Actor at Origin
#     pose = gymapi.Transform()
#     pose.p = gymapi.Vec3(0, 0, 0)
#     pose.r = gymapi.Quat(0, 0, 0, 1)
#     actor = gym.create_actor(env, asset, pose, "hand", 0, 1)

#     # Step simulation once to update transforms
#     gym.prepare_sim(sim)
#     for _ in range(10):
#         gym.simulate(sim)
#         gym.fetch_results(sim, True)
#     gym.refresh_rigid_body_state_tensor(sim)

#     # Get Body Names and Indices
#     body_names = gym.get_asset_rigid_body_names(asset)
#     body_dict = {name: i for i, name in enumerate(body_names)}

#     # Get Rigid Body States
#     # We need to use the tensor API or get_actor_rigid_body_states
#     # Since we are in a script, get_actor_rigid_body_states is easier
#     body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)

#     # Identify Knuckles (Base of fingers)
#     knuckles = [
#         "right_index_link_0",
#         "right_middle_link_0",
#         "right_ring_link_0",
#         "right_thumb_link_0",
#     ]

#     positions = []
#     print("--- Knuckle Positions (Relative to Root at 0,0,0) ---")
#     for name in knuckles:
#         if name in body_dict:
#             idx = body_dict[name]
#             pos = body_states[idx]["pose"]["p"]  # structured array
#             # pos is (x, y, z)
#             print(f"{name}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
#             positions.append([pos[0], pos[1], pos[2]])
#         else:
#             print(f"Warning: {name} not found")

#     if not positions:
#         print("No knuckles found.")
#         return

#     # Calculate Centroid
#     import numpy as np

#     positions = np.array(positions)
#     centroid = np.mean(positions, axis=0)

#     print("\n--- Calculated Palm Center (Centroid of Knuckles) ---")
#     print(f"X: {centroid[0]:.4f}")
#     print(f"Y: {centroid[1]:.4f}")
#     print(f"Z: {centroid[2]:.4f}")

#     print("\n--- Suggested Offset Correction ---")
#     print(f"To center the palm at (0,0,0), shift the hand root by:")
#     print(f"X: {-centroid[0]:.4f}")
#     print(f"Y: {-centroid[1]:.4f}")
#     print(f"Z: {-centroid[2]:.4f}")


# if __name__ == "__main__":
#     get_palm_offset()
