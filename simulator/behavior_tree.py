from enum import Enum, auto

import numpy as np

from simulator.trajectory_planner import TrapezoidalTrajectoryPlanner


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

    def _write_detection(self, blackboard, detection, depth):
        detection["name"] = self.object_name
        detection_2d = {
            "name": self.object_name,
            "camera": self.camera_name,
            "detected": bool(detection["detected"]),
            "center": None,
            "bbox": None,
        }

        detection_3d = {
            "name": self.object_name,
            "camera": self.camera_name,
            "detected": bool(detection["detected"]),
            "center": None,
        }

        x, y = np.asarray(detection["center"], dtype=np.float64).copy()
        cx = 180
        cy = 120

        fx = 360 / (2 * np.tan(45*np.pi / 360))
        fy = 240 / (2 * np.tan(45*np.pi / 360))

        Z = depth[int(y)][int(x)]
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy

        # print(X, Y, Z)        

        if detection["center"] is not None:
            detection_2d["center"] = np.asarray(detection["center"], dtype=np.float64).copy()
        if detection["bbox"] is not None:
            detection_2d["bbox"] = np.asarray(detection["bbox"], dtype=np.int64).copy()

        if detection["center"] is not None:
            detection_3d["center"] = np.asarray([X, Y, Z], dtype=np.float64).copy()

        blackboard["detected_object"] = {
            key: value for key, value in detection.items() if key != "mask"
        }
        blackboard["detected_object_2d"] = detection_2d
        blackboard["detected_object_3d"] = detection_3d

        objects_2d = blackboard.setdefault("detected_objects_2d", {})
        objects_2d[self.object_name] = detection_2d

        objects_3d = blackboard.setdefault("detected_objects_3d", {})
        objects_3d[self.object_name] = detection_3d

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
        depth = obs["depths"].get(self.camera_name)
        if image is None:
            return Status.FAILURE

        detection = self.detector(image)
        self._write_detection(blackboard, detection, depth)

        if detection["detected"]:
            return Status.SUCCESS

        return Status.RUNNING


class SolveIKNode(Node):
    def __init__(self, position, euler, output_key="goal_joints"):
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.euler = np.asarray(euler, dtype=np.float64).copy()
        self.output_key = output_key
        self.solved = False

    def reset(self):
        self.solved = False

    def tick(self, blackboard):
        if self.solved:
            return Status.SUCCESS

        obs = blackboard["obs"]
        kinematics = blackboard["kinematics"]
        current_joints = obs["state"]["joint_pos"][:6]

        c_pos, c_euler = kinematics.solve_fk(current_joints)

        obj_x, obj_y, obj_z = blackboard["detected_objects_3d"]["red_cube"]["center"]

        c_pos[1] += (obj_x)
        c_pos[0] += (obj_y - 0.1)
        c_pos[2] -= (obj_z-0.10)

        success, goal_joints = kinematics.solve_ik(
            c_pos,
            c_euler,
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


class CloseGripperNode(Node):
    def __init__(self, close_value=1.0):
        self.close_value = float(close_value)

    def tick(self, blackboard):
        blackboard["gripper_target"] = self.close_value
        blackboard["action"][-1] = self.close_value
        return Status.SUCCESS
