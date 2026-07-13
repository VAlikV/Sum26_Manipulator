from enum import Enum, auto

import mujoco
import numpy as np

from simulator.trajectory_planner import TrapezoidalTrajectoryPlanner


def mujoco_to_pin_position(position):
    position = np.asarray(position, dtype=np.float64).copy()
    position[0] *= -1.0
    position[1] *= -1.0
    return position


class Status(Enum):
    RUNNING = auto()
    SUCCESS = auto()
    FAILURE = auto()


class Node:
    def reset(self):
        pass

    def tick(self, blackboard):
        raise NotImplementedError


class Sequence(Node):
    def __init__(self, children):
        self.children = list(children)
        self.current_child = 0

    def reset(self):
        self.current_child = 0
        for child in self.children:
            child.reset()

    def tick(self, blackboard):
        while self.current_child < len(self.children):
            status = self.children[self.current_child].tick(blackboard)

            if status == Status.RUNNING:
                return Status.RUNNING
            if status == Status.FAILURE:
                return Status.FAILURE

            self.current_child += 1

        return Status.SUCCESS


class DetectObjectNode(Node):
    def __init__(self, object_name="object", camera_name=None, detector=None):
        self.object_name = object_name
        self.camera_name = camera_name
        self.detector = detector

    def _write_detection(self, blackboard, detection):
        detection["name"] = self.object_name
        detection_2d = {
            "name": self.object_name,
            "camera": self.camera_name,
            "detected": bool(detection["detected"]),
            "center": None,
            "bbox": None,
        }

        if detection["center"] is not None:
            detection_2d["center"] = np.asarray(detection["center"], dtype=np.float64).copy()
        if detection["bbox"] is not None:
            detection_2d["bbox"] = np.asarray(detection["bbox"], dtype=np.int64).copy()

        blackboard["detected_object"] = {
            key: value for key, value in detection.items() if key != "mask"
        }
        blackboard["detected_object_2d"] = detection_2d

        objects_2d = blackboard.setdefault("detected_objects_2d", {})
        objects_2d[self.object_name] = detection_2d

        if "mask" in detection:
            blackboard["detected_object_mask"] = detection["mask"]

    def tick(self, blackboard):
        if self.detector is None:
            detection = {
                "name": self.object_name,
                "detected": True,
                "center": None,
                "bbox": None,
                "area": 0.0,
            }
            self._write_detection(blackboard, detection)
            return Status.SUCCESS

        obs = blackboard["obs"]
        image = obs["images"].get(self.camera_name)
        if image is None:
            return Status.FAILURE

        detection = self.detector(image)
        self._write_detection(blackboard, detection)

        if detection["detected"]:
            return Status.SUCCESS

        return Status.RUNNING


class SolveIKNode(Node):
    def __init__(
        self,
        position=None,
        euler=None,
        object_name=None,
        camera_name=None,
        standoff=0.10,
        grasp_offset=None,
        output_key="goal_joints",
    ):
        self.position = None if position is None else np.asarray(position, dtype=np.float64).copy()
        self.euler = None if euler is None else np.asarray(euler, dtype=np.float64).copy()
        self.object_name = object_name
        self.camera_name = camera_name
        self.standoff = float(standoff)
        self.grasp_offset = np.array(
            [0.0, 0.0, -0.12] if grasp_offset is None else grasp_offset,
            dtype=np.float64,
        )
        self.output_key = output_key
        self.solved = False

    def reset(self):
        self.solved = False

    def _camera_id(self, env):
        camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if camera_id < 0:
            raise ValueError(f"Unknown camera: {self.camera_name}")
        return camera_id

    def _camera_intrinsics(self, env, camera_id, depth):
        height, width = depth.shape
        fovy = np.deg2rad(env.model.cam_fovy[camera_id])
        fy = height / (2.0 * np.tan(0.5 * fovy))
        fx = fy
        cx = 0.5 * (width - 1)
        cy = 0.5 * (height - 1)
        return fx, fy, cx, cy

    def _depth_at_object(self, depth, detection_2d, mask):
        if mask is not None:
            object_depth = depth[mask > 0]
            object_depth = object_depth[np.isfinite(object_depth) & (object_depth > 0.0)]
            if object_depth.size:
                return float(np.median(object_depth))

        u, v = np.asarray(detection_2d["center"], dtype=np.float64)
        u = int(np.clip(round(u), 0, depth.shape[1] - 1))
        v = int(np.clip(round(v), 0, depth.shape[0] - 1))

        patch = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
        patch = patch[np.isfinite(patch) & (patch > 0.0)]
        if not patch.size:
            return None
        return float(np.median(patch))

    def _camera_offset_from_xml(self, env, camera_id):
        camera_parent_body_id = env.model.cam_bodyid[camera_id]
        if camera_parent_body_id != env.ee_idx:
            raise ValueError(
                f"{self.camera_name} must be attached to {env.ee_name}, "
                f"but it is attached to body id {camera_parent_body_id}"
            )

        ee_to_camera_pos = env.model.cam_pos[camera_id].copy()

        ee_to_camera_quat = env.model.cam_quat[camera_id].copy()
        ee_to_camera_rot_flat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(ee_to_camera_rot_flat, ee_to_camera_quat)
        ee_to_camera_rot = ee_to_camera_rot_flat.reshape(3, 3)

        return ee_to_camera_pos, ee_to_camera_rot

    def _object_position_world(self, blackboard):
        env = blackboard["env"]
        obs = blackboard["obs"]

        detection_2d = blackboard.get("detected_objects_2d", {}).get(self.object_name)
        if detection_2d is None or not detection_2d["detected"] or detection_2d["center"] is None:
            return None

        depth = obs["depths"].get(self.camera_name)
        if depth is None:
            return None

        mask = blackboard.get("detected_object_mask")
        z = self._depth_at_object(depth, detection_2d, mask)
        if z is None:
            return None

        camera_id = self._camera_id(env)
        fx, fy, cx, cy = self._camera_intrinsics(env, camera_id, depth)

        u, v = np.asarray(detection_2d["center"], dtype=np.float64)
        x = (u - cx) * z / fx
        y = -(v - cy) * z / fy

        point_camera = np.array([x, y, -z], dtype=np.float64)
        camera_pos = env.data.cam_xpos[camera_id].copy()
        camera_rot = env.data.cam_xmat[camera_id].reshape(3, 3).copy()
        point_world = camera_pos + camera_rot @ point_camera
        point_pin = mujoco_to_pin_position(point_world)

        blackboard["detected_object_3d"] = {
            "name": self.object_name,
            "camera": self.camera_name,
            "detected": True,
            "center": point_world.copy(),
            "center_pin": point_pin.copy(),
            "depth": z,
        }
        blackboard.setdefault("detected_objects_3d", {})[self.object_name] = blackboard["detected_object_3d"]

        return point_world

    def _target_from_detected_object(self, blackboard, current_position, current_euler):
        env = blackboard["env"]
        point_world = self._object_position_world(blackboard)
        if point_world is None:
            return None, None

        camera_id = self._camera_id(env)
        ee_to_camera_pos, _ = self._camera_offset_from_xml(env, camera_id)

        target_euler = current_euler if self.euler is None else self.euler
        target_ee_rot_mujoco = env.data.xmat[env.ee_idx].reshape(3, 3).copy()

        approach_direction = target_ee_rot_mujoco @ np.array([0.0, 0.0, -1.0])
        target_grasp_pos = point_world - self.standoff * approach_direction
        target_position_mujoco = target_grasp_pos - target_ee_rot_mujoco @ self.grasp_offset
        target_position_pin = mujoco_to_pin_position(target_position_mujoco)

        blackboard["camera_offset_from_ee"] = ee_to_camera_pos.copy()
        blackboard["grasp_offset_from_ee"] = self.grasp_offset.copy()
        blackboard["grasp_target_position"] = target_grasp_pos.copy()
        blackboard["ik_target_position_mujoco"] = target_position_mujoco.copy()
        blackboard["ik_target_position"] = target_position_pin.copy()
        blackboard["ik_target_euler"] = target_euler.copy()
        return target_position_pin, target_euler

    def tick(self, blackboard):
        if self.solved:
            return Status.SUCCESS

        obs = blackboard["obs"]
        kinematics = blackboard["kinematics"]
        current_joints = obs["state"]["joint_pos"][:6]

        c_pos, c_euler = kinematics.solve_fk(current_joints)

        if self.object_name is None:
            target_position = self.position
            target_euler = c_euler if self.euler is None else self.euler
        else:
            target_position, target_euler = self._target_from_detected_object(
                blackboard,
                c_pos,
                c_euler,
            )

        if target_position is None or target_euler is None:
            return Status.FAILURE

        success, goal_joints = kinematics.solve_ik(
            target_position,
            target_euler,
            current_joints,
        )
        if not success:
            return Status.FAILURE

        blackboard[self.output_key] = goal_joints.copy()
        self.solved = True
        return Status.SUCCESS


class MovePoseNode(Node):
    def __init__(self, max_velocity, max_acceleration, goal_key="goal_joints"):
        self.max_velocity = np.asarray(max_velocity, dtype=np.float64).copy()
        self.max_acceleration = np.asarray(max_acceleration, dtype=np.float64).copy()
        self.goal_key = goal_key
        self.goal_joints = None
        self.planner = None
        self.t = 0.0

    def reset(self):
        self.goal_joints = None
        self.planner = None
        self.t = 0.0

    def _create_planner(self, blackboard):
        goal_joints = blackboard.get(self.goal_key)
        if goal_joints is None:
            return False

        obs = blackboard["obs"]
        self.goal_joints = np.asarray(goal_joints, dtype=np.float64).copy()
        start_joints = obs["state"]["joint_pos"][:len(self.goal_joints)]

        self.planner = TrapezoidalTrajectoryPlanner(
            start=start_joints,
            goal=self.goal_joints,
            max_velocity=self.max_velocity,
            max_acceleration=self.max_acceleration,
        )
        return True

    def tick(self, blackboard):
        env = blackboard["env"]
        action = blackboard["action"]

        if self.planner is None:
            if not self._create_planner(blackboard):
                return Status.FAILURE

        sample = self.planner.sample(self.t)
        if sample is None:
            action[:len(self.goal_joints)] = 0.0
            blackboard["joint_target"] = self.goal_joints.copy()
            return Status.SUCCESS

        joint_target, joint_velocity = sample
        action[:len(self.goal_joints)] = np.clip(
            joint_velocity,
            env.action_space.low[:len(self.goal_joints)],
            env.action_space.high[:len(self.goal_joints)],
        )
        blackboard["joint_target"] = joint_target.copy()

        self.t += env.sim_dt
        return Status.RUNNING


class MoveToFinalPoseNode(Node):
    def __init__(self, position_mujoco, max_velocity, max_acceleration, euler=None):
        self.position_mujoco = np.asarray(position_mujoco, dtype=np.float64).copy()
        self.max_velocity = np.asarray(max_velocity, dtype=np.float64).copy()
        self.max_acceleration = np.asarray(max_acceleration, dtype=np.float64).copy()
        self.euler = None if euler is None else np.asarray(euler, dtype=np.float64).copy()
        self.goal_joints = None
        self.planner = None
        self.t = 0.0

    def reset(self):
        self.goal_joints = None
        self.planner = None
        self.t = 0.0

    def _create_planner(self, blackboard):
        obs = blackboard["obs"]
        kinematics = blackboard["kinematics"]

        current_joints = obs["state"]["joint_pos"][:6]
        _, current_euler = kinematics.solve_fk(current_joints)

        target_position = mujoco_to_pin_position(self.position_mujoco)
        target_euler = current_euler if self.euler is None else self.euler

        success, goal_joints = kinematics.solve_ik(
            target_position,
            target_euler,
            current_joints,
        )
        if not success:
            return False

        self.goal_joints = goal_joints.copy()
        start_joints = obs["state"]["joint_pos"][:len(self.goal_joints)]

        self.planner = TrapezoidalTrajectoryPlanner(
            start=start_joints,
            goal=self.goal_joints,
            max_velocity=self.max_velocity,
            max_acceleration=self.max_acceleration,
        )

        blackboard["final_position_mujoco"] = self.position_mujoco.copy()
        blackboard["final_position_pin"] = target_position.copy()
        blackboard["final_goal_joints"] = self.goal_joints.copy()
        return True

    def tick(self, blackboard):
        env = blackboard["env"]
        action = blackboard["action"]

        if self.planner is None:
            if not self._create_planner(blackboard):
                return Status.FAILURE

        sample = self.planner.sample(self.t)
        if sample is None:
            action[:len(self.goal_joints)] = 0.0
            blackboard["joint_target"] = self.goal_joints.copy()
            return Status.SUCCESS

        joint_target, joint_velocity = sample
        action[:len(self.goal_joints)] = np.clip(
            joint_velocity,
            env.action_space.low[:len(self.goal_joints)],
            env.action_space.high[:len(self.goal_joints)],
        )
        blackboard["joint_target"] = joint_target.copy()

        self.t += env.sim_dt
        return Status.RUNNING


class CloseGripperNode(Node):
    def __init__(self, close_value=1.0):
        self.close_value = float(close_value)

    def tick(self, blackboard):
        blackboard["gripper_target"] = self.close_value
        blackboard["action"][-1] = self.close_value
        return Status.SUCCESS
