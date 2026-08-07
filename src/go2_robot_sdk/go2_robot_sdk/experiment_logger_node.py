import csv
import math
import os
import re
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger


class ExperimentLoggerNode(Node):
    """Record numeric performance and motion telemetry for each DWA run."""

    TIMESERIES_FIELDS = [
        "wall_time_iso",
        "elapsed_s",
        "cpu_percent",
        "gpu_percent",
        "ram_used_mb",
        "ram_total_mb",
        "detection_latency_ms",
        "detection_end_to_end_ms",
        "bev_latency_ms",
        "dwa_latency_ms",
        "requested_vx_mps",
        "requested_vy_mps",
        "requested_yaw_rate_radps",
        "safe_vx_mps",
        "safe_vy_mps",
        "safe_yaw_rate_radps",
        "measured_vx_mps",
        "measured_vy_mps",
        "measured_yaw_rad",
        "measured_yaw_rate_radps",
        "minimum_clearance_m",
        "dwa_status",
    ]

    def __init__(self):
        super().__init__("experiment_logger_node")
        self.declare_parameter(
            "output_directory", "~/go2_experiment_results"
        )
        self.declare_parameter("run_prefix", "run")
        self.declare_parameter("sample_rate", 5.0)
        self.declare_parameter("auto_start_on_goal", True)
        self.declare_parameter("goal_topic", "/dwa/goal")
        self.declare_parameter("status_topic", "/dwa/status")
        self.declare_parameter("requested_command_topic", "/dwa/cmd_vel_preview")
        self.declare_parameter("safe_command_topic", "/cmd_vel_safe_debug")
        self.declare_parameter("odom_topic", "/robot0/odom")
        self.declare_parameter(
            "clearance_topic", "/dwa/minimum_clearance"
        )

        self.output_directory = Path(
            os.path.expanduser(
                str(self.get_parameter("output_directory").value)
            )
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.run_prefix = str(self.get_parameter("run_prefix").value)
        self.auto_start_on_goal = bool(
            self.get_parameter("auto_start_on_goal").value
        )

        self.active = False
        self.run_id = ""
        self.run_start_monotonic = None
        self.run_start_wall = None
        self.last_status = "NOT_CONNECTED"
        self.seen_active_status = False
        self.latest = {
            "cpu_percent": math.nan,
            "gpu_percent": math.nan,
            "ram_used_mb": math.nan,
            "ram_total_mb": math.nan,
            "detection_latency_ms": math.nan,
            "detection_end_to_end_ms": math.nan,
            "bev_latency_ms": math.nan,
            "dwa_latency_ms": math.nan,
            "requested_vx_mps": 0.0,
            "requested_vy_mps": 0.0,
            "requested_yaw_rate_radps": 0.0,
            "safe_vx_mps": 0.0,
            "safe_vy_mps": 0.0,
            "safe_yaw_rate_radps": 0.0,
            "measured_vx_mps": 0.0,
            "measured_vy_mps": 0.0,
            "measured_yaw_rad": 0.0,
            "measured_yaw_rate_radps": 0.0,
            "minimum_clearance_m": math.nan,
        }
        self.samples = {
            "cpu_percent": [],
            "gpu_percent": [],
            "ram_used_mb": [],
            "detection_latency_ms": [],
            "detection_end_to_end_ms": [],
            "bev_latency_ms": [],
            "dwa_latency_ms": [],
            "minimum_clearance_m": [],
            "requested_vx_mps": [],
            "requested_yaw_rate_radps": [],
            "safe_vx_mps": [],
            "safe_yaw_rate_radps": [],
            "measured_vx_mps": [],
            "measured_yaw_rad": [],
            "measured_yaw_rate_radps": [],
        }
        self.timeseries_file = None
        self.events_file = None
        self.timeseries_writer = None
        self.events_writer = None
        self.last_cpu_snapshot = None
        self.tegrastats_process = None
        self.tegrastats_thread = None

        self.create_subscription(
            PoseStamped,
            self.get_parameter("goal_topic").value,
            self.goal_cb,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("status_topic").value,
            self.status_cb,
            10,
        )
        self.create_subscription(
            Twist,
            self.get_parameter("requested_command_topic").value,
            self.requested_command_cb,
            10,
        )
        self.create_subscription(
            Twist,
            self.get_parameter("safe_command_topic").value,
            self.safe_command_cb,
            10,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self.odom_cb,
            10,
        )
        self.create_subscription(
            Float32,
            self.get_parameter("clearance_topic").value,
            self.clearance_cb,
            10,
        )
        self.create_subscription(
            Float32,
            "/metrics/detection_latency_ms",
            lambda msg: self.metric_cb("detection_latency_ms", msg),
            10,
        )
        self.create_subscription(
            Float32,
            "/metrics/detection_end_to_end_ms",
            lambda msg: self.metric_cb(
                "detection_end_to_end_ms", msg
            ),
            10,
        )
        self.create_subscription(
            Float32,
            "/metrics/bev_latency_ms",
            lambda msg: self.metric_cb("bev_latency_ms", msg),
            10,
        )
        self.create_subscription(
            Float32,
            "/metrics/dwa_latency_ms",
            lambda msg: self.metric_cb("dwa_latency_ms", msg),
            10,
        )
        self.create_service(Trigger, "~/start_run", self.start_service_cb)
        self.create_service(Trigger, "~/stop_run", self.stop_service_cb)

        sample_rate = max(
            0.2, float(self.get_parameter("sample_rate").value)
        )
        self.create_timer(1.0 / sample_rate, self.sample_cb)
        self.start_tegrastats()
        self.get_logger().info(
            "Numeric experiment logger ready; output directory: "
            f"{self.output_directory}"
        )

    @staticmethod
    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)

    @staticmethod
    def yaw_from_quaternion(q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def start_tegrastats(self):
        try:
            self.tegrastats_process = subprocess.Popen(
                ["tegrastats", "--interval", "500"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, OSError) as exc:
            self.get_logger().warning(
                f"tegrastats unavailable; GPU usage will be blank: {exc}"
            )
            return
        self.tegrastats_thread = threading.Thread(
            target=self.read_tegrastats,
            daemon=True,
        )
        self.tegrastats_thread.start()

    def read_tegrastats(self):
        if self.tegrastats_process is None:
            return
        for line in self.tegrastats_process.stdout:
            gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
            ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
            if gpu_match:
                self.latest["gpu_percent"] = float(gpu_match.group(1))
            if ram_match:
                self.latest["ram_used_mb"] = float(ram_match.group(1))
                self.latest["ram_total_mb"] = float(ram_match.group(2))

    @staticmethod
    def read_cpu_snapshot():
        try:
            with open("/proc/stat", "r", encoding="utf-8") as stream:
                values = [
                    int(value)
                    for value in stream.readline().split()[1:]
                ]
        except (OSError, ValueError):
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    def update_cpu(self):
        snapshot = self.read_cpu_snapshot()
        if snapshot is None:
            return
        if self.last_cpu_snapshot is not None:
            total_delta = snapshot[0] - self.last_cpu_snapshot[0]
            idle_delta = snapshot[1] - self.last_cpu_snapshot[1]
            if total_delta > 0:
                self.latest["cpu_percent"] = (
                    100.0 * (total_delta - idle_delta) / total_delta
                )
        self.last_cpu_snapshot = snapshot

    def goal_cb(self, msg):
        if self.auto_start_on_goal:
            if self.active:
                self.finish_run("superseded_by_new_goal")
            self.start_run()
        self.log_event(
            "goal",
            f"frame={msg.header.frame_id},"
            f"x={msg.pose.position.x:.6f},"
            f"y={msg.pose.position.y:.6f}",
        )

    def status_cb(self, msg):
        previous = self.last_status
        self.last_status = msg.data
        if self.active and msg.data != previous:
            self.log_event("dwa_status", msg.data)
        if self.active and msg.data in ("PLANNING", "RECOVERY"):
            self.seen_active_status = True
        if (
            self.active
            and msg.data == "GOAL_REACHED"
            and (self.seen_active_status or self.elapsed() >= 0.5)
        ):
            self.finish_run("goal_reached")

    def requested_command_cb(self, msg):
        self.latest["requested_vx_mps"] = float(msg.linear.x)
        self.latest["requested_vy_mps"] = float(msg.linear.y)
        self.latest["requested_yaw_rate_radps"] = float(msg.angular.z)

    def safe_command_cb(self, msg):
        self.latest["safe_vx_mps"] = float(msg.linear.x)
        self.latest["safe_vy_mps"] = float(msg.linear.y)
        self.latest["safe_yaw_rate_radps"] = float(msg.angular.z)

    def odom_cb(self, msg):
        self.latest["measured_vx_mps"] = float(
            msg.twist.twist.linear.x
        )
        self.latest["measured_vy_mps"] = float(
            msg.twist.twist.linear.y
        )
        self.latest["measured_yaw_rad"] = self.yaw_from_quaternion(
            msg.pose.pose.orientation
        )
        self.latest["measured_yaw_rate_radps"] = float(
            msg.twist.twist.angular.z
        )

    def clearance_cb(self, msg):
        self.metric_cb("minimum_clearance_m", msg)

    def metric_cb(self, name, msg):
        value = float(msg.data)
        self.latest[name] = value
        if self.active and self.finite(value):
            self.samples[name].append(value)

    def start_service_cb(self, _request, response):
        if self.active:
            response.success = False
            response.message = f"Run already active: {self.run_id}"
            return response
        self.start_run()
        response.success = True
        response.message = f"Started {self.run_id}"
        return response

    def stop_service_cb(self, _request, response):
        if not self.active:
            response.success = False
            response.message = "No active run"
            return response
        run_id = self.run_id
        self.finish_run("manual_stop")
        response.success = True
        response.message = f"Stopped {run_id}"
        return response

    def start_run(self):
        if self.active:
            return
        timestamp = datetime.now().astimezone()
        self.run_id = (
            f"{self.run_prefix}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        )
        self.run_start_monotonic = time.monotonic()
        self.run_start_wall = timestamp
        self.seen_active_status = False
        for values in self.samples.values():
            values.clear()

        timeseries_path = (
            self.output_directory / f"{self.run_id}_timeseries.csv"
        )
        events_path = self.output_directory / f"{self.run_id}_events.csv"
        self.timeseries_file = open(
            timeseries_path, "w", newline="", encoding="utf-8"
        )
        self.events_file = open(
            events_path, "w", newline="", encoding="utf-8"
        )
        self.timeseries_writer = csv.DictWriter(
            self.timeseries_file,
            fieldnames=self.TIMESERIES_FIELDS,
        )
        self.timeseries_writer.writeheader()
        self.events_writer = csv.DictWriter(
            self.events_file,
            fieldnames=["wall_time_iso", "elapsed_s", "event", "value"],
        )
        self.events_writer.writeheader()
        self.active = True
        self.log_event("run_started", self.run_id)
        self.get_logger().info(f"Recording experiment {self.run_id}")

    def elapsed(self):
        if self.run_start_monotonic is None:
            return 0.0
        return time.monotonic() - self.run_start_monotonic

    def log_event(self, event, value):
        if not self.active or self.events_writer is None:
            return
        self.events_writer.writerow(
            {
                "wall_time_iso": datetime.now().astimezone().isoformat(),
                "elapsed_s": f"{self.elapsed():.6f}",
                "event": event,
                "value": value,
            }
        )
        self.events_file.flush()

    def sample_cb(self):
        self.update_cpu()
        if not self.active or self.timeseries_writer is None:
            return

        for name in (
            "cpu_percent",
            "gpu_percent",
            "ram_used_mb",
            "requested_vx_mps",
            "requested_yaw_rate_radps",
            "safe_vx_mps",
            "safe_yaw_rate_radps",
            "measured_vx_mps",
            "measured_yaw_rad",
            "measured_yaw_rate_radps",
        ):
            value = self.latest[name]
            if self.finite(value):
                self.samples[name].append(value)

        row = {
            "wall_time_iso": datetime.now().astimezone().isoformat(),
            "elapsed_s": f"{self.elapsed():.6f}",
            **{
                name: (
                    f"{value:.6f}" if self.finite(value) else ""
                )
                for name, value in self.latest.items()
            },
            "dwa_status": self.last_status,
        }
        self.timeseries_writer.writerow(row)
        self.timeseries_file.flush()

    @staticmethod
    def summarize(values):
        finite_values = [
            float(value) for value in values if math.isfinite(float(value))
        ]
        if not finite_values:
            return {"mean": "", "max": "", "p95": ""}
        ordered = sorted(finite_values)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return {
            "mean": statistics.fmean(finite_values),
            "max": max(finite_values),
            "p95": ordered[p95_index],
        }

    def finish_run(self, result):
        if not self.active:
            return
        completion_time = self.elapsed()
        self.log_event("run_finished", result)
        run_id = self.run_id
        summary_path = (
            self.output_directory / f"{run_id}_summary.csv"
        )

        summary = {
            "run_id": run_id,
            "start_time_iso": self.run_start_wall.isoformat(),
            "result": result,
            "completion_time_s": completion_time,
            "final_dwa_status": self.last_status,
        }
        for name, values in self.samples.items():
            stats = self.summarize(values)
            summary[f"mean_{name}"] = stats["mean"]
            summary[f"max_{name}"] = stats["max"]
            summary[f"p95_{name}"] = stats["p95"]
            finite_values = [
                abs(float(value))
                for value in values
                if math.isfinite(float(value))
            ]
            summary[f"peak_abs_{name}"] = (
                max(finite_values) if finite_values else ""
            )

        with open(
            summary_path, "w", newline="", encoding="utf-8"
        ) as summary_file:
            writer = csv.DictWriter(
                summary_file, fieldnames=list(summary.keys())
            )
            writer.writeheader()
            writer.writerow(summary)

        self.active = False
        if self.timeseries_file is not None:
            self.timeseries_file.close()
        if self.events_file is not None:
            self.events_file.close()
        self.timeseries_file = None
        self.events_file = None
        self.timeseries_writer = None
        self.events_writer = None
        self.get_logger().info(
            f"Finished {run_id}: result={result}, "
            f"duration={completion_time:.3f}s"
        )

    def destroy_node(self):
        if self.active:
            self.finish_run("aborted")
        if self.tegrastats_process is not None:
            self.tegrastats_process.terminate()
            try:
                self.tegrastats_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.tegrastats_process.kill()
        return super().destroy_node()


def main():
    rclpy.init()
    node = ExperimentLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
