import rclpy
from rclpy.node import Node
import tf2_geometry_msgs
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from tf2_geometry_msgs import PointStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from vesc_msgs.msg import VescImuStamped
from vesc_msgs.msg import VescStateStamped
import csv
import numpy as np
from visualization_msgs.msg import Marker
import tf2_ros
import time
import sys, os
sys.path.append(os.getcwd() + '/src')
from ws_params import raceline_path
import math
from ament_index_python.packages import get_package_share_directory


class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_pf_processed_inputs')

        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        print("Waiting for Buffer to be filled")
        time.sleep(3)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.wheel_base = 0.335
        self.max_steering_angle = 0.36
        self.speed_percentage = 1.0
        self.min_dist_point_reached = 0.7
        self.max_dist_point_reached = 1.3

        self.current_pose = None
        self.next_waypoint_index = None
        self.closest_waypoint_index = None
        self.target_speed = None
        self.max_curvature = None
        self.dist_point_reached = self.min_dist_point_reached

        # Load processed inputs
        #csv_path = os.path.join(get_package_share_directory('pp_nn_steering_node'),'data','processed_inputs.csv')
        #data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        #n_rows, n_cols = data.shape
        #window_size = (n_cols - 2) // 4
        #self.inputs = data[:, :window_size * 4].reshape(-1, window_size, 4)  # (N, 15, 4)
        #self.waypoints = data[:, -2:]  # Last two columns are x, y

        #self.get_logger().info(f"window_size: {window_size}")
        
        # === Raceline data ===
        raceline_data = np.genfromtxt(raceline_path, delimiter=";", comments='#')
        self.raceline_s = raceline_data[:, 0]
        self.raceline_x = raceline_data[:, 1]
        self.raceline_y = raceline_data[:, 2]
        self.raceline_kappa = raceline_data[:, 4]
        self.raceline_vx = raceline_data[:, 5]
        self.raceline_ax = raceline_data[:, 6]
        
        # ho fatto cosí per modificare la minor roba nel codice (non é pulitissimo)
        self.waypoints = np.array([self.raceline_x, self.raceline_y])
        
        # Calcola la curvatura massima (valore assoluto)
        self.max_curvature = np.max(np.abs(self.raceline_kappa))

        self.target_marker_pub = self.create_publisher(Marker, '/target_Marker', 5)
        self.target_marker = Marker()
        self.target_marker.ns = 'TargetVisualization'
        self.target_marker.id = 0
        self.target_marker.header.frame_id = 'map'
        self.target_marker.type = Marker.SPHERE
        self.target_marker.action = Marker.ADD
        self.target_marker.scale.x = 0.55
        self.target_marker.scale.y = 0.55
        self.target_marker.scale.z = 0.55
        self.target_marker.color.a = 1.0
        self.target_marker.color.r = 1.0

        self.pose_sub = self.create_subscription(PoseStamped, '/Car1/pose', self.pose_callback, 10)
        self.initialpose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initialpose_callback, 10)
        self.imu_sub = self.create_subscription(VescImuStamped, '/sensors/imu', self.imu_callback, 10)
        self.vesc_sub = self.create_subscription(VescStateStamped, '/sensors/core', self.vesc_callback, 10)

        self.vehicle_speed = 0.0
        self.vehicle_speed_initialized = False
        self.yaw_rate_measured = 0.0

    def p2p_dist(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def imu_callback(self, msg):
        yaw_rate_deg = msg.imu.angular_velocity.z
        self.yaw_rate_measured = yaw_rate_deg * np.pi / 180.0

    def initialpose_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.get_logger().info("Start index resetted")
        self.next_waypoint_index = None

    def pose_callback(self, msg):
        self.current_pose = msg.pose
        self.closest_waypoint_index = np.argmin(np.linalg.norm(self.waypoints - np.array([self.current_pose.position.x, self.current_pose.position.y]), axis=1))
        self.target_speed = self.raceline_vx[self.closest_waypoint_index]
        ackermann_cmd = self.calculate_control_commands()
        self.drive_pub.publish(ackermann_cmd)

    def vesc_callback(self, msg):
        self.vehicle_speed = msg.state.speed
        self.vehicle_speed_initialized = True
        #self.get_logger().info(f"Vehicle speed updated: {self.vehicle_speed}")

    def calculate_control_commands(self):
        ackermann_cmd = AckermannDriveStamped()

        if self.current_pose is not None and self.waypoints.any():
            if self.next_waypoint_index is None:
                self.set_initial_waypoint_index()

            next_wp = self.waypoints[self.next_waypoint_index]
            self.target_marker.pose.position.x = next_wp[0]
            self.target_marker.pose.position.y = next_wp[1]
            self.target_marker_pub.publish(self.target_marker)

            dx = next_wp[0] - self.current_pose.position.x
            dy = next_wp[1] - self.current_pose.position.y

            q = self.current_pose.orientation
            yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)[2]

            x_local = math.cos(yaw) * dx + math.sin(yaw) * dy
            y_local = -math.sin(yaw) * dx + math.cos(yaw) * dy

            L = math.sqrt(dx**2 + dy**2)
            if abs(y_local) > 1e-6:
                Radius = L**2 / (2 * abs(y_local))
                curvature = 1.0 / Radius
                steer = self.wheel_base / Radius
                if y_local < 0:
                    steer = -steer
                    curvature = -curvature
            else:
                steer = 0.0
                curvature = 0.0

            ackermann_cmd.drive.steering_angle = np.clip(steer, -self.max_steering_angle, self.max_steering_angle)
            self.calculate_point_reached_dist(ackermann_cmd.drive.steering_angle)
            ackermann_cmd.drive.speed = self.speed_percentage * self.target_speed

            if L < self.dist_point_reached:
                self.next_waypoint_index = (self.next_waypoint_index + 1) % len(self.waypoints)

            ackermann_cmd.header.stamp = self.get_clock().now().to_msg()

        return ackermann_cmd

    #def calculate_point_reached_dist(self, steering_angle):
    #    self.dist_point_reached = self.min_dist_point_reached + (1 - min(
    #        abs(steering_angle) / self.max_steering_angle +
    #        (abs(self.inputs[self.closest_waypoint_index][0][3]) + abs(self.inputs[self.next_waypoint_index][0][3])) / 2 / self.max_curvature,
    #        1)) * (self.max_dist_point_reached - self.min_dist_point_reached)
            
    def calculate_point_reached_dist(self, steering_angle):
        self.dist_point_reached = self.min_dist_point_reached + (1 - min(abs(steering_angle) / self.max_steering_angle + (abs(self.raceline_data[self.closest_waypoint_index, 4]) + abs(self.raceline_data[self.next_waypoint_index, 4])) / 2 / self.max_curvature, 1)) \
            * (self.max_dist_point_reached - self.min_dist_point_reached)

    def set_initial_waypoint_index(self):
        if self.current_pose:
            q = self.current_pose.orientation
            yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)[2]

            min_distance = float('inf')
            closest_index = None
            for i, wp in enumerate(self.waypoints):
                dx = wp[0] - self.current_pose.position.x
                dy = wp[1] - self.current_pose.position.y

                x_local = math.cos(yaw) * dx + math.sin(yaw) * dy
                y_local = -math.sin(yaw) * dx + math.cos(yaw) * dy

                if x_local > 1.0 and abs(math.atan2(y_local, x_local)) < self.max_steering_angle:
                    dist = x_local**2 + y_local**2
                    if dist < min_distance:
                        min_distance = dist
                        closest_index = i

            if closest_index is None:
                raise ValueError('No suitable initial waypoint found')

            self.next_waypoint_index = closest_index

    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_controller = PurePursuitController()
    rclpy.spin(pure_pursuit_controller)
    pure_pursuit_controller.destroy_node()
    pure_pursuit_controller.curvature_log_file.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
