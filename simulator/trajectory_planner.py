import numpy as np


class CubicTrajectoryPlanner:
    def __init__(self, start, goal, duration):
        self.set_trajectory(start, goal, duration)

    def set_trajectory(self, start, goal, duration):
        self.start = np.asarray(start, dtype=np.float64).copy()
        self.goal = np.asarray(goal, dtype=np.float64).copy()
        self.duration = float(duration)

        if self.duration <= 0.0:
            raise ValueError("duration must be positive")
        if self.start.shape != self.goal.shape:
            raise ValueError("start and goal must have the same shape")

        tf = self.duration
        T = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [tf**3, tf**2, tf, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [3.0 * tf**2, 2.0 * tf, 1.0, 0.0],
        ], dtype=np.float64)

        boundary = np.vstack([
            self.start,
            self.goal,
            np.zeros_like(self.start),
            np.zeros_like(self.start),
        ])

        self.coeff = np.linalg.solve(T, boundary)

    def update(self, start, goal, duration):
        self.set_trajectory(start, goal, duration)

    def sample(self, t):
        t = float(t)
        if t > self.duration:
            return None

        t = max(t, 0.0)
        a3, a2, a1, a0 = self.coeff

        position = a3 * t**3 + a2 * t**2 + a1 * t + a0
        velocity = 3.0 * a3 * t**2 + 2.0 * a2 * t + a1

        return position.copy(), velocity.copy()

class TrapezoidalTrajectoryPlanner:
    def __init__(self, start, goal, max_velocity, max_acceleration):
        self.set_trajectory(start, goal, max_velocity, max_acceleration)

    def set_trajectory(self, start, goal, max_velocity, max_acceleration):
        self.start = np.asarray(start, dtype=np.float64).copy()
        self.goal = np.asarray(goal, dtype=np.float64).copy()
        self.max_velocity = np.asarray(max_velocity, dtype=np.float64)
        self.max_acceleration = np.asarray(max_acceleration, dtype=np.float64)

        if self.start.shape != self.goal.shape:
            raise ValueError("start and goal must have the same shape")

        self.max_velocity = np.broadcast_to(self.max_velocity, self.start.shape).copy()
        self.max_acceleration = np.broadcast_to(self.max_acceleration, self.start.shape).copy()

        if np.any(self.max_velocity <= 0.0):
            raise ValueError("max_velocity must be positive")
        if np.any(self.max_acceleration <= 0.0):
            raise ValueError("max_acceleration must be positive")

        self.delta = self.goal - self.start
        self.direction = np.sign(self.delta)
        self.distance = np.abs(self.delta)

        min_duration = self._calc_min_duration()
        self.duration = float(np.max(min_duration))

        self.accel_time = np.zeros_like(self.distance)
        self.cruise_time = np.zeros_like(self.distance)
        self.peak_velocity = np.zeros_like(self.distance)
        self.acceleration = np.zeros_like(self.distance)

        active = self.distance > 0.0
        if not np.any(active):
            self.duration = 0.0
            return

        for idx in np.where(active)[0]:
            d = self.distance[idx]
            a_max = self.max_acceleration[idx]

            root = self.duration**2 - 4.0 * d / a_max
            root = max(root, 0.0)

            ta = 0.5 * (self.duration - np.sqrt(root))
            v = a_max * ta

            if v > self.max_velocity[idx]:
                v = self.max_velocity[idx]
                ta = v / a_max

            self.accel_time[idx] = ta
            self.cruise_time[idx] = self.duration - 2.0 * ta
            self.peak_velocity[idx] = v
            self.acceleration[idx] = self.direction[idx] * a_max

    def _calc_min_duration(self):
        duration = np.zeros_like(self.distance)

        for idx, d in enumerate(self.distance):
            if d == 0.0:
                continue

            v_max = self.max_velocity[idx]
            a_max = self.max_acceleration[idx]
            t_accel = v_max / a_max
            d_accel = 0.5 * a_max * t_accel**2

            if d <= 2.0 * d_accel:
                duration[idx] = 2.0 * np.sqrt(d / a_max)
            else:
                duration[idx] = 2.0 * t_accel + (d - 2.0 * d_accel) / v_max

        return duration

    def update(self, start, goal, max_velocity, max_acceleration):
        self.set_trajectory(start, goal, max_velocity, max_acceleration)

    def sample(self, t):
        t = float(t)
        if t > self.duration:
            return None

        t = max(t, 0.0)
        position = self.start.copy()
        velocity = np.zeros_like(self.start)

        for idx, d in enumerate(self.distance):
            if d == 0.0:
                continue

            ta = self.accel_time[idx]
            tc = self.cruise_time[idx]
            v = self.direction[idx] * self.peak_velocity[idx]
            a = self.acceleration[idx]

            if t < ta:
                q = self.start[idx] + 0.5 * a * t**2
                dq = a * t
            elif t < ta + tc:
                q_accel = 0.5 * a * ta**2
                q = self.start[idx] + q_accel + v * (t - ta)
                dq = v
            else:
                t_dec = t - ta - tc
                q_accel = 0.5 * a * ta**2
                q_cruise = v * tc
                q = self.start[idx] + q_accel + q_cruise + v * t_dec - 0.5 * a * t_dec**2
                dq = v - a * t_dec

            position[idx] = q
            velocity[idx] = dq

        return position.copy(), velocity.copy()
