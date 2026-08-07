"""A simple DWA-style simulation that uses the planner class."""

import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

try:
    from .dwa_planner import DWAPlanner
except ImportError:  # pragma: no cover - allows running the file directly
    from dwa_planner import DWAPlanner


def _update_state(x, y, yaw, linear_velocity, angular_velocity, dt):
    """Advance the robot state with a simple kinematic model."""
    next_yaw = yaw + angular_velocity * dt
    next_x = x + linear_velocity * math.cos(next_yaw) * dt
    next_y = y + linear_velocity * math.sin(next_yaw) * dt
    return next_x, next_y, next_yaw


def main(
    goal_x=0.0,
    goal_y=3.0,
    robot_length=0.7,
    robot_width=0.35,
    max_linear_acceleration=0.2,
    max_angular_acceleration=40.0 * math.pi / 180.0,
    dt=0.1,
    max_steps=600,
    goal_tolerance=0.5,
    show_animation=True,
):
    """Run the standalone DWA test and return its termination metrics."""
    planner = DWAPlanner()
    planner.clear_obstacles()

    goal = (goal_x, goal_y)
    # Deterministic test: one obstacle halfway between the robot and goal.
    obstacle_specs = [(0.0, 0.0, 0.5)]
    for obstacle_x, obstacle_y, radius in obstacle_specs:
        planner.add_obstacles(obstacle_x, obstacle_y, radius)

    robot_specs = {
        "max_linear_speed": 1.0,
        "min_linear_speed": -0.5,
        "max_angular_speed": 40.0 * math.pi / 180.0,
        "max_linear_acceleration": max_linear_acceleration,
        "max_angular_acceleration": max_angular_acceleration,
        "v_resolution": 0.01,
        "yaw_rate_resolution": 0.1 * math.pi / 180.0,
        "robot_length": robot_length,
        "robot_width": robot_width,
        "safety_margin": 0.1,
        "predict_time": 3.0,
    }
    simulation_specs = {
        "to_goal_cost_gain": 0.15,
        "speed_cost_gain": 1.0,
        # Collision remains a hard rejection in DWAPlanner. This lower soft
        # cost prevents distant, already-safe obstacles from dominating the
        # heading and speed terms.
        "obstacle_cost_gain": 0.35,
        "robot_stuck_flag_cons": 0.05,
        "stuck_turn_steps": 200,
        "stuck_detection_steps": 1,
        "recovery_turn_angle": 0.7,
        # Move decisively during the bypass phase instead of crawling.
        "recovery_forward_speed": 0.25,
        "recovery_bypass_distance": 1.2,
    }

    # Begin below and close to the obstacle, facing upward toward the goal.
    # This preserves the previous 1.05 m robot-to-obstacle separation.
    x, y, yaw = 0.0, -1.05, math.pi / 2.0
    linear_velocity = 0.0
    angular_velocity = 0.0
    history = [(x, y)]
    status = "max_steps"
    minimum_clearance = float("inf")
    steps_run = 0

    if show_animation:
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.ion()

    for step in range(max_steps):
        steps_run = step + 1
        action = planner.get_robot_next_action(
            robot_specs,
            {
                "position": {"x": x, "y": y},
                "yaw": yaw,
                "linear_velocity": linear_velocity,
                "angular_velocity": angular_velocity,
            },
            {"x": goal[0], "y": goal[1]},
            simulation_specs=simulation_specs,
            dt=dt,
        )

        linear_velocity = action["linear_velocity"]
        angular_velocity = action["angular_velocity"]
        x, y, yaw = _update_state(x, y, yaw, linear_velocity, angular_velocity, dt)
        history.append((x, y))

        state = [x, y, yaw, linear_velocity, angular_velocity]
        clearance = planner.state_clearance(
            state, robot_specs, simulation_specs=simulation_specs, dt=dt
        )
        minimum_clearance = min(minimum_clearance, clearance)
        collision = clearance <= 0.0
        if collision:
            status = "collision"

        if show_animation:
            ax.clear()
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.0, 4.0)
            ax.set_aspect("equal")
            ax.grid(True)

            for obstacle in planner.obstacles:
                circle = plt.Circle((obstacle["x"], obstacle["y"]), obstacle["r"], color="k", alpha=0.6)
                ax.add_patch(circle)

            ax.plot(goal[0], goal[1], "bo", markersize=8)
            ax.plot([point[0] for point in history], [point[1] for point in history], color="tab:green", linewidth=2)
            ax.plot(x, y, "ro", markersize=6)
            half_length = robot_length / 2.0
            half_width = robot_width / 2.0
            local_corners = [
                (half_length, half_width),
                (half_length, -half_width),
                (-half_length, -half_width),
                (-half_length, half_width),
            ]
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            world_corners = [
                (
                    x + cosine * corner_x - sine * corner_y,
                    y + sine * corner_x + cosine * corner_y,
                )
                for corner_x, corner_y in local_corners
            ]
            footprint = Polygon(
                world_corners, fill=False, color="tab:red", linewidth=2
            )
            ax.add_patch(footprint)

            dx = 0.4 * math.cos(yaw)
            dy = 0.4 * math.sin(yaw)
            ax.arrow(x - dx, y - dy, dx, dy, head_width=0.15, head_length=0.2, fc="r", ec="r")
            mode = (
                f"RECOVERY {action.get('recovery_steps', 0)}"
                if action.get("recovery_active")
                else "NORMAL"
            )
            ax.set_title(
                "DWA simulation "
                f"[{mode}] v={linear_velocity:.2f} m/s "
                f"w={angular_velocity:.2f} rad/s"
            )
            plt.pause(0.001)

        if collision:
            break

        if math.hypot(x - goal[0], y - goal[1]) < goal_tolerance:
            status = "goal_reached"
            break

    if show_animation:
        plt.ioff()
        plt.show()

    result = {
        "status": status,
        "obstacles": [dict(obstacle) for obstacle in planner.obstacles],
        "steps": steps_run,
        "completion_time_s": steps_run * dt,
        "path_length_m": sum(
            math.hypot(
                history[index][0] - history[index - 1][0],
                history[index][1] - history[index - 1][1],
            )
            for index in range(1, len(history))
        ),
        "final_x": x,
        "final_y": y,
        "distance_to_goal": math.hypot(x - goal[0], y - goal[1]),
        "minimum_clearance": minimum_clearance,
    }
    print(
        "DWA simulation: "
        f"status={result['status']}, "
        f"obstacles={len(result['obstacles'])}, steps={result['steps']}, "
        f"path_length={result['path_length_m']:.3f} m, "
        f"distance_to_goal={result['distance_to_goal']:.3f} m, "
        f"minimum_clearance={result['minimum_clearance']:.3f} m"
    )
    return result


if __name__ == "__main__":
    main()
