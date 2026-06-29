import numpy as np
from simulator.env import Env, PinKinematics, get_robot_xml_path
import matplotlib.pyplot as plt
import cv2
import time

kinematics = PinKinematics(model_path=get_robot_xml_path("ur10e2f85.xml"), ee_name="gripper_base")

env = Env(xml_path="scene.xml",
            sim_timestep = 0.001,
            control_hz = 10.0,
            mode = "realtime",   # "realtime" | "fast
            control_mode="joint_position",  # "joint_position" | "joint_velocity"
            joint_velocity_limit=0.5,
            max_episode_steps = 1000,
            render_mode="all",   # None | "human" | "rgb_array" | "all"
)

obs, info = env.reset()

start_joints = obs["state"]["joint_pos"].copy()
joints = obs["state"]["joint_pos"].copy()
start_position, start_euler = kinematics.solve_fk(start_joints)

print(start_joints)
print(start_position, start_euler)

# action = np.zeros(env.action_dim)
# action[-1] = 0.0

t = time.time()

# plt.ion()
# fig, axes = plt.subplots(1, 3, figsize=(10, 5))
position = start_position.copy()
euler = start_euler.copy()

position[0] += 0.2
position[1] -= 0.2
position[2] -= 0.15

t = time.time()

s, r_joints = kinematics.solve_ik(start_position, start_euler, joints)
joints[0:6] = r_joints

for _ in range(10001):


    if (time.time() - t >= 5):
        s, r_joints = kinematics.solve_ik(position, euler, joints)
        joints[0:6] = r_joints
        # t = time.time() + 10000000

    print(s)
    print(joints, r_joints)
    print(start_position)

    # action[:] = 0.0
    # action[0] = 0.2 * np.sin(_ / (10 * np.pi))
    # action[1] = 0.1 * np.cos(_ / (10 * np.pi))

    obs, reward, terminated, truncated, info = env.step(joints)

    imgs = obs["images"]
    joint_pos = obs["state"]["joint_pos"]

    pos, euler = kinematics.solve_fk(joint_pos)

    # print("FK pos:", pos, euler)
    # print("Sim pos:", obs["state"]["ee_pos"], obs["state"]["ee_euler"])

    # for ax, (name, img) in zip(axes, imgs.items()):
    #     ax.clear()
    #     ax.imshow(img)
    #     ax.set_title(name)
    #     ax.axis("off")

    # plt.pause(0.001)



    # print("POS:", obs["state"]["joint_vel"])
    # print("POS:", obs["state"]["ee_pos"])
    # print("EULER:", obs["state"]["ee_euler"])
    # print("LIN_VEL:",obs["state"]["ee_lin_vel"])
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
