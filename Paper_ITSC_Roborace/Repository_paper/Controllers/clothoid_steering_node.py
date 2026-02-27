import rclpy
from rclpy.node import Node
import tf2_geometry_msgs
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from tf2_geometry_msgs import PointStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from vesc_msgs.msg import VescImuStamped, VescStateStamped
import csv
import numpy as np
from visualization_msgs.msg import Marker
import tf2_ros
import time
import sys, os
import math
sys.path.append(os.getcwd() + '/src')
from ws_params import raceline_path

# Import Clothoid library
import Clothoids


class ClothoidController(Node):
    def __init__(self):
        super().__init__('clothoid_steering_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        print("Waiting for Buffer to be filled")
        time.sleep(3)

        # === Publishers and Subscribers ===
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/Car1/pose', self.pose_callback, 10)
        self.initalpose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.intialpose_callback, 10)
        self.imu_sub = self.create_subscription(VescImuStamped, '/sensors/imu', self.imu_callback, 10)
        self.core_sub = self.create_subscription(VescStateStamped, '/sensors/core', self.core_callback, 10)
        self.target_marker_pub = self.create_publisher(Marker, '/target_Marker', 5)

        # === Parameters ===
        self.wheel_base = 0.335
        self.max_steering_angle = 0.36
        self.speed_percentage = 1.0
        self.min_dist_point_reached = 1.3
        self.max_dist_point_reached = 1.8

        self.current_pose = None
        self.next_waypoint_index = None
        self.closest_waypoint_index = None
        self.target_speed = None
        self.target_accel = None
        self.max_curvature = None
        self.dist_point_reached = self.min_dist_point_reached
        self.yaw_rate_measured = 0.0
        self.measured_speed = 0.0

        # === Raceline data ===
        self.raceline_data = np.genfromtxt(raceline_path, delimiter=";", comments='#')
        self.s_array = self.raceline_data[:, 0]   # curvilinea
        self.x_array = self.raceline_data[:, 1]
        self.y_array = self.raceline_data[:, 2]
        self.psi_array = self.raceline_data[:, 7]
        self.kappa_array = self.raceline_data[:, 4]
        self.vx_array = self.raceline_data[:, 5]
        self.ax_array = self.raceline_data[:, 6]

        # === ClothoidList ===
        self.curve_list = Clothoids.ClothoidList("curve_list")
        self.curve_list.build_G1(self.x_array, self.y_array)
        self.curve_list.make_closed()
        self.max_curvature = np.max(abs(self.kappa_array))
        self.curve = Clothoids.G2solve3arc()

        # === Marker setup ===
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


    # -------------------------------
    # Callbacks
    # -------------------------------
    
    def imu_callback(self, msg):
        yaw_rate_deg = msg.imu.angular_velocity.z
        self.yaw_rate_measured = yaw_rate_deg * np.pi / 180.0  # rad/s
        
    def core_callback(self, msg):
        self.measured_speed = msg.state.speed / 4244.0      # m/s
    
    def intialpose_callback(self, msg):
        self.get_logger().info("Start index resetted")
        self.next_waypoint_index = None

    def pose_callback(self, msg):
        self.current_pose = msg.pose

        # Closest point on clothoid list 
        x_ego = self.current_pose.position.x
        y_ego = self.current_pose.position.y
        s_ego, _ = self.curve_list.findST1(x_ego, y_ego)
        
        self.s_ego = s_ego

        # Interpola velocità target da raceline
        self.target_speed = np.interp(s_ego, self.s_array, self.vx_array) 

        # Generate control command
        ackermann_cmd = self.calculate_control_commands(s_ego)
        self.drive_pub.publish(ackermann_cmd)
         

    def calculate_control_commands(self, s_ego):
        ackermann_cmd = AckermannDriveStamped()

        if not self.current_pose:
            return ackermann_cmd

        x_ego = self.current_pose.position.x
        y_ego = self.current_pose.position.y
        quat = self.current_pose.orientation
        yaw_ego = self.euler_from_quaternion(quat.x, quat.y, quat.z, quat.w)[2]
        kappa_ego = self.yaw_rate_measured / max(self.measured_speed, 0.1)     # (rad/s)/(m/s) = 1/m

        # Dynamic lookahead point
        s_lookahead = s_ego + self.dist_point_reached
        x_lookahead, y_lookahead = self.curve_list.eval(s_lookahead)
        theta_lookahead = self.curve_list.theta(s_lookahead)
        kappa_lookahead = self.curve_list.theta_D(s_lookahead)
        
        # Marker
        self.target_marker.pose.position.x = x_lookahead
        self.target_marker.pose.position.y = y_lookahead
        self.target_marker_pub.publish(self.target_marker)
        
        # Clothoid generation from ego to lookahead
        self.curve.build(x_ego, y_ego, yaw_ego, kappa_ego, x_lookahead, y_lookahead, theta_lookahead, kappa_lookahead)
        
        s_values = np.arange(0, self.curve.total_length(), 0.05, dtype=np.float64)
        points = np.zeros((s_values.size, 2))
        
        for i in range(s_values.size):
            points[i,:] = self.curve.eval(s_values[i])

        # Evaluate curvature 
        self.kappa = self.curve.theta_D(0.0)
        
        # Steering command
        steer = np.clip(self.kappa * self.wheel_base, -self.max_steering_angle, self.max_steering_angle)
        ackermann_cmd.drive.steering_angle = steer

        # Dynamic point reached distance
        self.calculate_point_reached_dist(steer)

        # Speed command
        ackermann_cmd.drive.speed = self.target_speed
        ackermann_cmd.header.stamp = self.get_clock().now().to_msg()

        return ackermann_cmd


    # Function to calculate the dynamic lookahead distance based on steering angle and curvature
    def calculate_point_reached_dist(self, steering_angle):
        # stessa logica che avevi, solo adattata
        self.dist_point_reached = self.min_dist_point_reached + (1 - min(
            abs(steering_angle) / self.max_steering_angle +
            np.mean(abs(self.kappa_array)) / self.max_curvature,
            1
        )) * (self.max_dist_point_reached - self.min_dist_point_reached)


    # Function to convert quaternion to euler angles
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
    node = ClothoidController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
