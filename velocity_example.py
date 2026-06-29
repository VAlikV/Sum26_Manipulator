import numpy as np
from simulator.env import Env, PinKinematics, get_robot_xml_path
import matplotlib.pyplot as plt
import cv2
import time

kinematics = PinKinematics(model_path=get_robot_xml_path("ur10e2f85.xml"), ee_name="gripper_base")

env = Env(xml_path="scene.xml",
            sim_timestep = 0.001,
            control_hz = 100.0,
            mode = "realtime",   # "realtime" | "fast
            control_mode="joint_velocity",  # "joint_position" | "joint_velocity"
            joint_velocity_limit=0.5,
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

# plt.ion()
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))

t = time.time()

for _ in range(30000):

    velocity[0] = 0.15 * np.cos( _ / (10*np.pi))
    velocity[2] = -0.15 * np.sin( _ / (10*np.pi))

    J = kinematics.solve_jacobian(joint_pos)

    dq = np.linalg.pinv(J) @ velocity
    all_joints[0:6] = dq

    obs, reward, terminated, truncated, info = env.step(all_joints)

    images = obs["images"]
    joint_pos = obs["state"]["joint_pos"][0:6]
    gripper_pos = obs["state"]["joint_pos"][6]

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

# plt.ioff()
# plt.show()
