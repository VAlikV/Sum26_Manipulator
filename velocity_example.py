import numpy as np
from simulator.env import Env, PinKinematics, get_robot_xml_path
import matplotlib.pyplot as plt
import cv2
import time

def calcA(tf, q_s, q_f):

    T = np.array([[0, 0, 0, 1],
                  [tf**3, tf**2, tf, 1],
                  [0, 0, 1, 0],
                  [3*tf**2, 2*tf, 1, 0]], dtype=float)
    
    q = np.array([[q_s],
                  [q_f],
                  [0],
                  [0]], dtype=float)
    
    a = np.linalg.inv(T) @ q

    return a

kinematics = PinKinematics(model_path=get_robot_xml_path("ur10e2f85.xml"), ee_name="gripper_base")

env = Env(xml_path="scene.xml",
            sim_timestep = 0.001,
            control_hz = 100.0,
            mode = "realtime",   # "realtime" | "fast
            control_mode="joint_velocity",  # "joint_position" | "joint_velocity"
            joint_velocity_limit=0.4,
            max_episode_steps = -1,
            render_mode="all",   # None | "human" | "rgb_array" | "all"
)

obs, info = env.reset()

start_joints = obs["state"]["joint_pos"].copy()
all_joints = np.zeros(7)

joint_pos = obs["state"]["joint_pos"][0:6]
gripper_pos = obs["state"]["joint_pos"][6]

start_position, start_euler = kinematics.solve_fk(start_joints)

position = start_position.copy()
euler = start_euler.copy()

velocity = np.zeros(6)
joint_target = joint_pos.copy()
dq = np.zeros(6)

target_position_graph = []
current_position_graph = []
joints_graph = []
joints_target_graph = []
joints_vel_graph = []
joints_target_vel_graph = []

s_q0 = start_joints[0]
s_q1 = start_joints[1]

t_q0 = s_q0 + 0.35
t_q1 = s_q1 - 0.15

tf = 1.0
a0 = calcA(tf, s_q0, t_q0)
a1 = calcA(tf, s_q1, t_q1)

# plt.ion()
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))

t = time.time()

trajectory_steps = int(np.ceil(tf / env.sim_dt)) + 1

for i in range(trajectory_steps):

    velocity[0] = 0.15 * np.sin( i / (10*np.pi))
    velocity[2] = -0.15 * np.sin( i / (10*np.pi))
    # position += velocity[0:3] * env.sim_dt

    # t_sec = min(i * env.sim_dt, tf)
    
    # joint_target[0] = a0[0, 0] * t_sec**3 + a0[1, 0] * t_sec**2 + a0[2, 0] * t_sec + a0[3, 0]
    # joint_target[1] = a1[0, 0] * t_sec**3 + a1[1, 0] * t_sec**2 + a1[2, 0] * t_sec + a1[3, 0]

    # dq[:] = 0.0
    # dq[0] = 3*a0[0, 0] * t_sec**2 + 2*a0[1, 0] * t_sec + a0[2, 0]
    # dq[1] = 3*a1[0, 0] * t_sec**2 + 2*a1[1, 0] * t_sec + a1[2, 0]

    J = kinematics.solve_jacobian(joint_pos)
    dq = np.linalg.pinv(J) @ velocity

    dq_target = np.clip(dq, env.action_space.low[0:6], env.action_space.high[0:6])
    all_joints[0:6] = dq_target

    obs, reward, terminated, truncated, info = env.step(all_joints)

    images = obs["images"]
    joint_pos = obs["state"]["joint_pos"][0:6]
    gripper_pos = obs["state"]["joint_pos"][6]
    current_position, current_euler = kinematics.solve_fk(joint_pos)
    position, euler = kinematics.solve_fk(joint_target)

    target_position_graph.append(position.copy())
    current_position_graph.append(current_position.copy())
    joints_graph.append(joint_pos.copy())
    joints_target_graph.append(joint_target.copy())
    joints_vel_graph.append(obs["state"]["joint_vel"][0:6].copy())
    joints_target_vel_graph.append(dq_target.copy())

    # for ax, (name, img) in zip(axes, imgs.items()):
    #     ax.clear()
    #     ax.imshow(img)
    #     ax.set_title(name)
    #     ax.axis("off")

    # plt.pause(0.001)

    if terminated or truncated:
        print("Episode ended:", terminated, truncated, info)
        obs, info = env.reset()

        print("Время:", time.time() - t)

env.close()

target_position_graph = np.asarray(target_position_graph)
current_position_graph = np.asarray(current_position_graph)
joints_graph = np.asarray(joints_graph)
joints_target_graph = np.asarray(joints_target_graph)
joints_vel_graph = np.asarray(joints_vel_graph)
joints_target_vel_graph = np.asarray(joints_target_vel_graph)
steps = np.arange(len(target_position_graph))

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
for axis_id, axis_name in enumerate(["x", "y", "z"]):
    axes[axis_id].plot(steps, current_position_graph[:, axis_id], label=f"${axis_name}_c$")
    axes[axis_id].plot(steps, target_position_graph[:, axis_id], "--", label=f"${axis_name}_t$")
    axes[axis_id].set_ylabel(f"{axis_name}, m")
    axes[axis_id].grid(True)
    axes[axis_id].legend()

axes[-1].set_xlabel("step")
fig.tight_layout()

joint_names = ["q1", "q2", "q3", "q4", "q5", "q6"]

fig_joints, axes_joints = plt.subplots(3, 2, sharex=True, figsize=(10, 8))
axes_joints = axes_joints.ravel()
for joint_id, joint_name in enumerate(joint_names):
    axes_joints[joint_id].plot(steps, joints_graph[:, joint_id], label=f"${joint_name}_c$")
    axes_joints[joint_id].plot(steps, joints_target_graph[:, joint_id], "--", label=f"${joint_name}_t$")
    axes_joints[joint_id].set_ylabel(f"{joint_name}, rad")
    axes_joints[joint_id].grid(True)
    axes_joints[joint_id].legend()

axes_joints[-1].set_xlabel("step")
axes_joints[-2].set_xlabel("step")
fig_joints.suptitle("Joint Positions")
fig_joints.tight_layout()

fig_vel, axes_vel = plt.subplots(3, 2, sharex=True, figsize=(10, 8))
axes_vel = axes_vel.ravel()
for joint_id, joint_name in enumerate(joint_names):
    axes_vel[joint_id].plot(steps, joints_vel_graph[:, joint_id], label=f"$\\dot{{{joint_name}}}_c$")
    axes_vel[joint_id].plot(steps, joints_target_vel_graph[:, joint_id], "--", label=f"$\\dot{{{joint_name}}}_t$")
    axes_vel[joint_id].set_ylabel(f"{joint_name}, rad/s")
    axes_vel[joint_id].grid(True)
    axes_vel[joint_id].legend()

axes_vel[-1].set_xlabel("step")
axes_vel[-2].set_xlabel("step")
fig_vel.suptitle("Joint Velocities")
fig_vel.tight_layout()
# plt.ioff()
plt.show()
