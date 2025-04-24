import time
import os
from isaacgym import gymapi, gymtorch, gymutil
import torch
import numpy as np
np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# def calculate_jacobian()

threshold = 1e-3
damping = 0.05
alpha = 0.5
target_position = torch.tensor([.5,0.5,0.5], device=device)

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.rest_offset = 0.0
sim_params.physx.contact_offset = 0.001
sim_params.physx.friction_offset_threshold = 0.001
sim_params.physx.friction_correlation_distance = 0.0005
sim_params.physx.num_threads = 4
sim_params.physx.use_gpu = True

device_id = 0         
graphics_id = 0

sim = gym.create_sim(device_id, graphics_id, gymapi.SIM_PHYSX, sim_params)
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# workspace bounds for the env
spacing = 1.0
lower   = gymapi.Vec3(-spacing, -spacing, 0.0)
upper   = gymapi.Vec3(spacing,  spacing, spacing)
env     = gym.create_env(sim, lower, upper, 1)

asset_root = "assets"
asset_file = "franka_description/robots/franka_panda.urdf"

print(f"asset root: {asset_root}, file; {asset_file}")

asset_opts = gymapi.AssetOptions()
asset_opts.fix_base_link = True
asset_opts.flip_visual_attachments = True
asset_opts.disable_gravity = True
asset_opts.thickness = 0.001
asset_opts.armature = 0.01
franka_asset = gym.load_asset(sim, asset_root, asset_file, asset_opts)

franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = torch.from_numpy(franka_dof_props["lower"]).to(device=device)
franka_upper_limits = torch.from_numpy(franka_dof_props["upper"]).to(device=device)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
pose.r = gymapi.Quat(0, 0, 0, 1)

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

franka_handle = gym.create_actor(env, franka_asset, pose, "franka", 0, 1)

gym.viewer_camera_look_at(viewer, None,
                          gymapi.Vec3( 1.5, 1.0, 1.0),  
                          gymapi.Vec3( 0.0, 0.0, 0.7))  #target

gym.prepare_sim(sim)

franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
franka_hand_index = franka_link_dict["panda_hand"]

# Set position control for the joints
for i in range(7):
    franka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
    franka_dof_props["stiffness"][i] = 400.0
    franka_dof_props["damping"][i] = 40.0
gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

#pos
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

#angles
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

dof_pos = dof_states[:, 0]  # Current positions
_dof_position_targets = torch.zeros_like(dof_pos)
dof_position_targets = gymtorch.unwrap_tensor(_dof_position_targets)

while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensor for latest data
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    # print(f"End effector position: {hand_pos}")

    current_joint_pos = dof_states[:7, 0]
    current_pos = rb_states[franka_hand_index, :3]

    delta_x = target_position - current_pos
    error_norm = torch.norm(delta_x)

    if error_norm < threshold:
        print("Target reached!")

    # Get the Jacobian for the end effector (position part only)
    J = jacobian[0, franka_hand_index, :3, :7]
    JT = J.transpose(0, 1)
    JJT = J @ JT
    reg = torch.eye(JJT.shape[0], device=J.device) * (damping ** 2)
    J_pinv = JT @ torch.inverse(JJT + reg)

    # delta_theta = alpha * JT @ delta_x
    delta_theta = alpha * J_pinv @ delta_x
    
    # Update joint positions
    new_joint_pos = current_joint_pos + delta_theta
    new_joint_pos = torch.clamp(new_joint_pos, franka_lower_limits[:7], franka_upper_limits[:7])

    _dof_position_targets[:7] = new_joint_pos
    gym.set_dof_position_target_tensor(sim, dof_position_targets)

    print(f"Error: {error_norm.item():.6f}")
    
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    time.sleep(1.0/60.0) #60Hz     

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)