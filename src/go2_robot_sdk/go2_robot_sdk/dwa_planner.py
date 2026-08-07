import math


class DWAPlanner:
    """Original-style DWA planner with an oriented rectangular footprint."""

    def __init__(self):
        self.obstacles = []
        self.stuck_turn_steps_remaining = 0
        self.stuck_turn_sign = None
        self.stalled_cycles = 0
        self.recovery_anchor = None
        self.recovery_start_yaw = None
        self.recovery_phase = None

    def add_obstacles(self, x, y, r):
        """Add a circular obstacle at ROS ground-plane position (x, y)."""
        self.obstacles.append({"x": float(x), "y": float(y), "r": float(r)})

    def clear_obstacles(self):
        self.obstacles.clear()

    @staticmethod
    def _value(primary, secondary, key, default):
        return float(primary.get(key, secondary.get(key, default)))

    def _config(self, robot_specs, simulation_specs, dt):
        sim = simulation_specs or {}
        return {
            "max_speed": self._value(robot_specs, sim, "max_linear_speed", 0.5),
            "min_speed": self._value(robot_specs, sim, "min_linear_speed", 0.0),
            "max_yaw_rate": self._value(
                robot_specs, sim, "max_angular_speed", 0.6
            ),
            "max_accel": self._value(
                robot_specs, sim, "max_linear_acceleration", 0.5
            ),
            "max_yaw_accel": self._value(
                robot_specs, sim, "max_angular_acceleration", 1.0
            ),
            "v_resolution": self._value(
                robot_specs, sim, "v_resolution", 0.05
            ),
            "yaw_resolution": self._value(
                robot_specs, sim, "yaw_rate_resolution", 0.05
            ),
            "robot_length": self._value(
                robot_specs, sim, "robot_length", 0.7
            ),
            "robot_width": self._value(
                robot_specs, sim, "robot_width", 0.35
            ),
            "safety_margin": self._value(
                robot_specs, sim, "safety_margin", 0.1
            ),
            "predict_time": self._value(
                robot_specs, sim, "predict_time", 3.0
            ),
            "goal_cost_gain": self._value(
                sim, {}, "to_goal_cost_gain", 0.15
            ),
            "speed_cost_gain": self._value(
                sim, {}, "speed_cost_gain", 1.0
            ),
            "obstacle_cost_gain": self._value(
                sim, {}, "obstacle_cost_gain", 0.5
            ),
            "stuck_threshold": self._value(
                sim, {}, "robot_stuck_flag_cons", 0.05
            ),
            "stuck_turn_steps": max(
                1, int(self._value(sim, {}, "stuck_turn_steps", 12))
            ),
            "stuck_detection_steps": max(
                1, int(self._value(
                    sim, {}, "stuck_detection_steps", 4
                ))
            ),
            "recovery_turn_angle": max(
                0.0, self._value(
                    sim, {}, "recovery_turn_angle", 0.7
                )
            ),
            "recovery_forward_speed": max(
                0.0, self._value(
                    sim, {}, "recovery_forward_speed", 0.08
                )
            ),
            "recovery_bypass_distance": max(
                0.0, self._value(
                    sim, {}, "recovery_bypass_distance", 1.2
                )
            ),
            "dt": float(dt),
        }

    @staticmethod
    def _parse_robot_state(robot_position):
        if robot_position is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        if isinstance(robot_position, dict):
            position = robot_position.get("position", robot_position)
            return [
                float(position.get("x", 0.0)),
                float(position.get("y", 0.0)),
                float(robot_position.get("yaw", 0.0)),
                float(robot_position.get("linear_velocity", 0.0)),
                float(robot_position.get("angular_velocity", 0.0)),
            ]
        values = list(robot_position)
        values.extend([0.0] * (5 - len(values)))
        return [float(value) for value in values[:5]]

    @staticmethod
    def _parse_goal(goal_position):
        if isinstance(goal_position, dict):
            return (
                float(goal_position.get("x", 0.0)),
                float(goal_position.get("y", 0.0)),
            )
        return float(goal_position[0]), float(goal_position[1])

    @staticmethod
    def _motion(state, linear_velocity, angular_velocity, dt):
        x, y, yaw, _, _ = state
        yaw += angular_velocity * dt
        x += linear_velocity * math.cos(yaw) * dt
        y += linear_velocity * math.sin(yaw) * dt
        return [x, y, yaw, linear_velocity, angular_velocity]

    @staticmethod
    def _samples(lower, upper, resolution):
        if resolution <= 0.0:
            raise ValueError("DWA velocity resolutions must be positive")
        if upper < lower:
            return []
        count = int(math.floor((upper - lower) / resolution))
        values = [lower + index * resolution for index in range(count + 1)]
        if not values or upper - values[-1] > 1e-9:
            values.append(upper)
        if lower <= 0.0 <= upper:
            values.append(0.0)
        return sorted(set(round(value, 12) for value in values))

    @staticmethod
    def _dynamic_window(state, config):
        current_v = state[3]
        current_w = state[4]
        dt = config["dt"]
        return (
            max(
                config["min_speed"],
                current_v - config["max_accel"] * dt,
            ),
            min(
                config["max_speed"],
                current_v + config["max_accel"] * dt,
            ),
            max(
                -config["max_yaw_rate"],
                current_w - config["max_yaw_accel"] * dt,
            ),
            min(
                config["max_yaw_rate"],
                current_w + config["max_yaw_accel"] * dt,
            ),
        )

    def _predict_trajectory(self, initial_state, v, w, config):
        state = list(initial_state)
        trajectory = [list(state)]
        elapsed = 0.0
        while elapsed < config["predict_time"] - 1e-9:
            state = self._motion(state, v, w, config["dt"])
            trajectory.append(state)
            elapsed += config["dt"]
        return trajectory

    @staticmethod
    def _point_to_rectangle_clearance(state, obstacle, config):
        """Distance from a circular obstacle to the oriented robot rectangle."""
        dx = obstacle["x"] - state[0]
        dy = obstacle["y"] - state[1]
        cosine = math.cos(state[2])
        sine = math.sin(state[2])

        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        outside_x = max(
            abs(local_x) - config["robot_length"] / 2.0, 0.0
        )
        outside_y = max(
            abs(local_y) - config["robot_width"] / 2.0, 0.0
        )
        center_to_rectangle = math.hypot(outside_x, outside_y)
        return (
            center_to_rectangle
            - obstacle["r"]
            - config["safety_margin"]
        )

    def trajectory_clearance(self, trajectory, config):
        if not self.obstacles:
            return float("inf")
        return min(
            self._point_to_rectangle_clearance(state, obstacle, config)
            for state in trajectory
            for obstacle in self.obstacles
        )

    def trajectory_center_distance(self, trajectory):
        """Minimum robot-center to obstacle-boundary distance."""
        if not self.obstacles:
            return float("inf")
        return min(
            math.hypot(
                state[0] - obstacle["x"],
                state[1] - obstacle["y"],
            )
            - obstacle["r"]
            for state in trajectory
            for obstacle in self.obstacles
        )

    def state_in_collision(
        self, state, robot_specs, simulation_specs=None, dt=0.2
    ):
        config = self._config(robot_specs, simulation_specs, dt)
        return self.trajectory_clearance([state], config) <= 0.0

    def state_clearance(
        self, state, robot_specs, simulation_specs=None, dt=0.2
    ):
        config = self._config(robot_specs, simulation_specs, dt)
        return self.trajectory_clearance([state], config)

    @staticmethod
    def _goal_heading_cost(trajectory, goal):
        final_state = trajectory[-1]
        goal_angle = math.atan2(
            goal[1] - final_state[1],
            goal[0] - final_state[0],
        )
        heading_error = goal_angle - final_state[2]
        return abs(
            math.atan2(
                math.sin(heading_error),
                math.cos(heading_error),
            )
        )

    def _obstacle_cost(self, trajectory, config):
        """Original-style inverse-distance cost with footprint rejection."""
        clearance = self.trajectory_clearance(trajectory, config)
        if clearance <= 0.0:
            return float("inf"), clearance
        if math.isinf(clearance):
            return 0.0, clearance

        # Match the reference DWA: the rectangular footprint determines
        # collision, while ordinary obstacle cost uses inverse distance.
        center_distance = self.trajectory_center_distance(trajectory)
        return 1.0 / max(center_distance, 1e-6), clearance

    def _select_bypass_sign(self, state):
        """Choose a persistent turn direction away from the nearest obstacle."""
        if not self.obstacles:
            return -1.0

        cosine = math.cos(state[2])
        sine = math.sin(state[2])
        nearest = None
        for obstacle in self.obstacles:
            dx = obstacle["x"] - state[0]
            dy = obstacle["y"] - state[1]
            local_x = cosine * dx + sine * dy
            local_y = -sine * dx + cosine * dy
            if local_x < 0.0:
                continue
            distance = math.hypot(local_x, local_y)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, local_y)

        if nearest is None:
            return -1.0
        # Positive local y is left of the robot, so turn right, and vice versa.
        return -1.0 if nearest[1] >= 0.0 else 1.0

    def get_robot_next_action(
        self,
        robot_specs,
        robot_position,
        goal_position,
        simulation_specs=None,
        dt=0.2,
    ):
        """Return the lowest-cost collision-free DWA command."""
        state = self._parse_robot_state(robot_position)
        goal = self._parse_goal(goal_position)
        config = self._config(robot_specs, simulation_specs, dt)

        if (
            self.recovery_anchor is not None
            and self.recovery_phase == "bypass"
        ):
            progress = math.hypot(
                state[0] - self.recovery_anchor[0],
                state[1] - self.recovery_anchor[1],
            )
            if progress >= config["recovery_bypass_distance"]:
                self.stuck_turn_steps_remaining = 0
                self.stuck_turn_sign = None
                self.stalled_cycles = 0
                self.recovery_anchor = None
                self.recovery_start_yaw = None
                self.recovery_phase = None

        dynamic_window = self._dynamic_window(state, config)
        v_samples = self._samples(
            dynamic_window[0],
            dynamic_window[1],
            config["v_resolution"],
        )
        w_samples = self._samples(
            dynamic_window[2],
            dynamic_window[3],
            config["yaw_resolution"],
        )

        best_action = None
        minimum_cost = float("inf")

        for linear_velocity in v_samples:
            for angular_velocity in w_samples:
                trajectory = self._predict_trajectory(
                    state,
                    linear_velocity,
                    angular_velocity,
                    config,
                )
                obstacle_cost, clearance = self._obstacle_cost(
                    trajectory, config
                )
                if math.isinf(obstacle_cost):
                    continue

                heading_cost = (
                    config["goal_cost_gain"]
                    * self._goal_heading_cost(trajectory, goal)
                )
                speed_cost = (
                    config["speed_cost_gain"]
                    * (config["max_speed"] - trajectory[-1][3])
                )
                weighted_obstacle_cost = (
                    config["obstacle_cost_gain"] * obstacle_cost
                )

                final_cost = (
                    heading_cost + speed_cost + weighted_obstacle_cost
                )

                if final_cost <= minimum_cost:
                    minimum_cost = final_cost
                    best_action = {
                        "linear_velocity": linear_velocity,
                        "angular_velocity": angular_velocity,
                        "cost": final_cost,
                        "heading_cost": heading_cost,
                        "distance_cost": 0.0,
                        "speed_cost": speed_cost,
                        "obstacle_cost": weighted_obstacle_cost,
                        "yaw_change_cost": 0.0,
                        "trajectory": trajectory,
                        "minimum_clearance": clearance,
                        "braking_clearance": clearance,
                        "dynamic_window": dynamic_window,
                        "recovery_active": False,
                        "recovery_steps": 0,
                        "recovery_yaw_change": 0.0,
                    }

        stalled = (
            best_action is not None
            and abs(best_action["linear_velocity"])
            <= config["stuck_threshold"]
            and abs(state[3]) <= config["stuck_threshold"]
        )
        if self.stuck_turn_steps_remaining == 0:
            if stalled:
                self.stalled_cycles += 1
            else:
                self.stalled_cycles = 0
        else:
            self.stalled_cycles = 0

        if (
            self.stalled_cycles >= config["stuck_detection_steps"]
            and self.stuck_turn_steps_remaining == 0
        ):
            if self.stuck_turn_sign is None:
                self.stuck_turn_sign = self._select_bypass_sign(state)
            if self.recovery_anchor is None:
                self.recovery_anchor = (state[0], state[1])
                self.recovery_start_yaw = state[2]
                self.recovery_phase = "turn"
            self.stuck_turn_steps_remaining = config["stuck_turn_steps"]
            self.stalled_cycles = 0

        if self.stuck_turn_steps_remaining > 0 and best_action is not None:
            yaw_change = abs(
                math.atan2(
                    math.sin(state[2] - self.recovery_start_yaw),
                    math.cos(state[2] - self.recovery_start_yaw),
                )
            )
            if (
                self.recovery_phase == "turn"
                and yaw_change >= config["recovery_turn_angle"]
            ):
                self.recovery_phase = "bypass"
                self.recovery_anchor = (state[0], state[1])

            if self.recovery_phase == "turn":
                candidate_rates = [
                    rate
                    for rate in (dynamic_window[2], dynamic_window[3])
                    if rate * self.stuck_turn_sign > 1e-9
                ]
            else:
                candidate_rates = list(w_samples)

            if candidate_rates:
                if self.recovery_phase == "turn":
                    stuck_turn_rate = max(candidate_rates, key=abs)
                else:
                    # Straighten during the bypass instead of continuing to
                    # curl away from the goal.
                    stuck_turn_rate = min(candidate_rates, key=abs)
                recovery_step = (
                    config["stuck_turn_steps"]
                    - self.stuck_turn_steps_remaining
                    + 1
                )
                recovery_velocity = 0.0
                if self.recovery_phase == "bypass":
                    recovery_velocity = min(
                        config["recovery_forward_speed"],
                        dynamic_window[1],
                    )
                    recovery_velocity = max(
                        dynamic_window[0], recovery_velocity
                    )
                turn_trajectory = self._predict_trajectory(
                    state, recovery_velocity, stuck_turn_rate, config
                )
                turn_obstacle_cost, turn_clearance = self._obstacle_cost(
                    turn_trajectory, config
                )
                if (
                    math.isinf(turn_obstacle_cost)
                    and recovery_velocity > 0.0
                ):
                    recovery_velocity = 0.0
                    turn_candidates = [
                        rate
                        for rate in (dynamic_window[2], dynamic_window[3])
                        if rate * self.stuck_turn_sign > 1e-9
                    ]
                    if turn_candidates:
                        stuck_turn_rate = max(turn_candidates, key=abs)
                    turn_trajectory = self._predict_trajectory(
                        state, 0.0, stuck_turn_rate, config
                    )
                    turn_obstacle_cost, turn_clearance = self._obstacle_cost(
                        turn_trajectory, config
                    )
                if not math.isinf(turn_obstacle_cost):
                    heading_cost = (
                        config["goal_cost_gain"]
                        * self._goal_heading_cost(turn_trajectory, goal)
                    )
                    speed_cost = (
                        config["speed_cost_gain"] * config["max_speed"]
                    )
                    weighted_obstacle_cost = (
                        config["obstacle_cost_gain"] * turn_obstacle_cost
                    )
                    best_action.update(
                        {
                            "linear_velocity": recovery_velocity,
                            "angular_velocity": stuck_turn_rate,
                            "cost": (
                                heading_cost
                                + speed_cost
                                + weighted_obstacle_cost
                            ),
                            "heading_cost": heading_cost,
                            "speed_cost": speed_cost,
                            "obstacle_cost": weighted_obstacle_cost,
                            "trajectory": turn_trajectory,
                            "minimum_clearance": turn_clearance,
                            "braking_clearance": turn_clearance,
                            "recovery_active": True,
                            "recovery_steps": recovery_step,
                        }
                    )
                    if self.stuck_turn_steps_remaining > 1:
                        self.stuck_turn_steps_remaining -= 1
                else:
                    self.stuck_turn_steps_remaining = 0
            else:
                self.stuck_turn_steps_remaining = 0

        if best_action is not None:
            return best_action

        stop_trajectory = self._predict_trajectory(
            state, 0.0, 0.0, config
        )
        clearance = self.trajectory_clearance(stop_trajectory, config)
        return {
            "linear_velocity": 0.0,
            "angular_velocity": 0.0,
            "cost": float("inf"),
            "heading_cost": float("inf"),
            "distance_cost": 0.0,
            "speed_cost": config["speed_cost_gain"] * config["max_speed"],
            "obstacle_cost": float("inf"),
            "yaw_change_cost": 0.0,
            "trajectory": stop_trajectory,
            "minimum_clearance": clearance,
            "braking_clearance": clearance,
            "dynamic_window": dynamic_window,
            "recovery_active": False,
            "recovery_steps": 0,
            "recovery_yaw_change": 0.0,
        }
