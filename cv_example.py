import numpy as np
from simulator.env import Env, PinKinematics, get_robot_xml_path
from simulator.behavior_tree import (
    CloseGripperNode,
    DetectObjectNode,
    MoveToFinalPoseNode,
    MovePoseNode,
    Sequence,
    SolveIKNode,
)
from simulator.red_cube_detector import detect_red_cube, draw_detection
import matplotlib.pyplot as plt
import time


def detect_red_cube_rgb(image):
    return detect_red_cube(image, min_area=40, input_format="rgb")


kinematics = PinKinematics(model_path=get_robot_xml_path("ur10e2f85.xml"), ee_name="gripper_base")

env = Env(xml_path="scene.xml",
            sim_timestep = 0.001,
            control_hz = 100.0,
            mode = "realtime",   # "realtime" | "fast
            control_mode="joint_velocity",  # "joint_position" | "joint_velocity"
            joint_velocity_limit=0.9,
            max_episode_steps = -1,
            render_mode="all",   # None | "human" | "rgb_array" | "all"
)

obs, info = env.reset()

max_velocity = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
max_acceleration = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
detector_camera = "cam_gripper"
final_position_mujoco = np.array([-0.2, -0.65, 0.5])

tree = Sequence([
    DetectObjectNode(
        object_name="red_cube",
        camera_name=detector_camera,
        detector=detect_red_cube_rgb,
    ),
    SolveIKNode(
        object_name="red_cube",
        camera_name=detector_camera,
        standoff=0.0,
        grasp_offset=np.array([0.0, 0.0, 0.12]),
    ),
    MovePoseNode(
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
    ),
    CloseGripperNode(close_value=1.0),
    # MoveToFinalPoseNode(
    #     position_mujoco=final_position_mujoco,
    #     max_velocity=max_velocity,
    #     max_acceleration=max_acceleration,
    # ),
    # CloseGripperNode(close_value=0.0),
])

blackboard = {
    "env": env,
    "kinematics": kinematics,
    "obs": obs,
    "action": np.zeros(env.action_dim),
    "gripper_target": 0.0,
}

# plt.ion()
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

t = time.time()
steps = int(np.ceil(10.0 / env.sim_dt)) + 1

for _ in range(steps):
    blackboard["action"][:] = 0.0
    blackboard["action"][-1] = blackboard["gripper_target"]

    tree.tick(blackboard)

    obs, reward, terminated, truncated, info = env.step(blackboard["action"])
    blackboard["obs"] = obs

    images = obs["images"]
    depths = obs["depths"]

    # rgb_ax, depth_ax = axes
    # rgb_ax.clear()
    # depth_ax.clear()

    # if detector_camera in images:
    #     img = images[detector_camera]
    #     if blackboard.get("detected_object") is not None:
    #         img = draw_detection(img, blackboard["detected_object"])

    #     rgb_ax.imshow(img)
    #     detection = blackboard.get("detected_object")
    #     if detection is not None:
    #         rgb_ax.set_title(f"{detector_camera} RGB: {detection['detected']} area={detection['area']:.0f}")
    #     else:
    #         rgb_ax.set_title(f"{detector_camera} RGB")
    # else:
    #     rgb_ax.set_title(f"{detector_camera} RGB missing")
    # rgb_ax.axis("off")

    # if detector_camera in depths:
    #     depth = depths[detector_camera]
    #     finite_depth = depth[np.isfinite(depth)]
    #     if finite_depth.size:
    #         vmin = np.percentile(finite_depth, 5)
    #         vmax = np.percentile(finite_depth, 95)
    #         if vmax <= vmin:
    #             vmax = vmin + 1e-6
    #     else:
    #         vmin = 0.0
    #         vmax = 1.0

    #     depth_ax.imshow(depth, cmap="magma", vmin=vmin, vmax=vmax)
    #     depth_ax.set_title(f"{detector_camera} depth")
    # else:
    #     depth_ax.set_title(f"{detector_camera} depth missing")
    # depth_ax.axis("off")

    # plt.pause(0.001)

    if terminated or truncated:
        print("Episode ended:", terminated, truncated, info)
        obs, info = env.reset()
        blackboard["obs"] = obs
        tree.reset()

        print("Время:", time.time() - t)

env.close()

# plt.ioff()
# plt.show()
