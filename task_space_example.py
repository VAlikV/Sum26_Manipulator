import numpy as np
from simulator.env import AssemblingEnv
import matplotlib.pyplot as plt
import cv2
import time

env = AssemblingEnv(xml_path="scene.xml",
            sim_timestep = 0.001,
            control_hz = 20.0,
            mode = "realtime",   # "realtime" | "fast"
            control_mode="joint_velocity",  # "joint_position" | "joint_velocity"
            joint_velocity_limit=0.5,
            max_episode_steps = 1000,
            render_mode="all",   # None | "human" | "rgb_array" | "all"
)

obs, info = env.reset()

action = np.zeros(env.action_dim)
action[-1] = 0.0

t = time.time()

# plt.ion()
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))

for _ in range(1001):
    action[:] = 0.0
    # action[0] = 0.2 * np.sin(_ / (10 * np.pi))
    # action[1] = 0.1 * np.cos(_ / (10 * np.pi))

    obs, reward, terminated, truncated, info = env.step(action)

    imgs = obs["images"]

    # for ax, (name, img) in zip(axes, imgs.items()):
    #     ax.clear()
    #     ax.imshow(img)
    #     ax.set_title(name)
    #     ax.axis("off")

    # plt.pause(0.001)

    # print("POS:", obs["state"]["joint_vel"])
    # print("POS:", obs["state"]["ee_pos"])
    # print("EULER:", obs["state"]["ee_euler"])
    print("LIN_VEL:",obs["state"]["ee_lin_vel"])
    # print("ANG_VEL:",obs["state"]["ee_ang_vel"])
    # print("JOINTS:",obs["state"]["joint_pos"])
    # print("OBJECTS:")
    # for k in obs["objects"].keys():
    #     print(k, obs["objects"][k]["lin_vel"])
    # print()

    if terminated or truncated:
        print("Episode ended:", terminated, truncated, info)
        obs, info = env.reset()

        print("Время:", time.time() - t)

    # if _ % 100 == 0:
    #     obs, info = env.reset()

env.close()

# plt.ioff()
# plt.show()
