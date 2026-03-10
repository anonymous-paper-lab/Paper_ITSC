import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Imu
from vesc_msgs.msg import VescImuStamped, VescStateStamped
import numpy as np
import torch
from tf_transformations import euler_from_quaternion
import Clothoids
import math
import tf2_ros
import time
import sys, os
sys.path.append(os.getcwd() + '/src')
from ws_params import raceline_path
import pygigi
import csv


from pp_nn_steering_node.steer_controller_base_onnx import TracerModel


class PPNNController(Node):
    def __init__(self):
        super().__init__('pp_nn_steering_node')

        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        print("Waiting for Buffer to be filled")
        time.sleep(3)

        # === Publishers and Subscribers ===
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/Car1/pose', self.pose_callback, 10)
        self.initialpose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initialpose_callback, 10)
        self.imu_sub = self.create_subscription(VescImuStamped, '/sensors/imu', self.imu_callback, 10)
        self.core_sub = self.create_subscription(VescStateStamped, '/sensors/core', self.core_callback, 10)
        self.target_marker_pub = self.create_publisher(Marker, '/target_Marker', 5)

        # === Parameters ===
        self.wheel_base = 0.335
        self.max_steering_angle = 0.36
        self.speed_percentage = 1.0
        self.min_dist_point_reached = 1.1    #0.8 #1.1 #1.4
        self.max_dist_point_reached = 1.7   #1.4 #1.7 #2.0

        self.current_pose = None
        self.target_speed = None
        self.dist_point_reached = self.min_dist_point_reached
        self.yaw_rate_measured = 0.0
        self.measured_speed = 0.0 

        # === Raceline data ===
        raceline_data = np.genfromtxt(raceline_path, delimiter=";", comments='#')
        self.raceline_s = raceline_data[:, 0]
        self.raceline_x = raceline_data[:, 1]
        self.raceline_y = raceline_data[:, 2]
        self.raceline_kappa = raceline_data[:, 4]
        self.raceline_vx = raceline_data[:, 5]
        self.raceline_ax = raceline_data[:, 6]

        # === ClothoidList ===
        self.curve_list = Clothoids.ClothoidList("curve_list")
        self.curve_list.build_G1(self.raceline_x, self.raceline_y)
        self.curve_list.make_closed()
        
        seval = np.linspace(0,self.curve_list.length(),500)
        keval = [self.curve_list.theta_D(s) for s in seval]
        self.kappa_max_road = np.max(np.abs(keval))

        # === Marker setup ===
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

        # === NN model ===
        self.model = TracerModel()
        self.model.eval()
        
        # === Clothoid model ===
        self.curve = Clothoids.ClothoidCurve('curve')
        self.s_ego = 0.0

        # -------------------------------
        # Integration with GIGI for velocity replanning
        # -------------------------------
        self.use_gigi = True                    # Flag to activate/disactivate Velocity replanner
        self.vx_init = 0.0                      # Initial velocity for GIGI optimization (updated at each iteration with la velocitá misurata del VESC)
        self.delta_t_s = 0.05                   # Time step for GIGI output grid (20Hz)
        self.num_points = 6                     # Number of points in the NN input sequence 
        self.num_NN_points = self.num_points
        self.gigi_delay = 0.1                   # Estimated delay of the GIGI optimization and actuation pipeline (in seconds) - used to select the velocity from the GIGI profile to send to the vehicle
        
        # Timing for GIGI replanning
        self.last_replan_time = self.get_clock().now()

        # === GGV Diagramm Custom ===
        vx_points = np.linspace(0, 6, 7)                         
        ay_max_val = 6.0
        ay_points = np.linspace(-ay_max_val, ay_max_val, 11)

        # Dynamic limits for ax based on velocity (elliptical limits in the ay-ax plane)
        ax_max_low_v = 4.0      # ax_max at zero velocity 
        ax_max_high_v = 3.0     # ax_max at max velocity
        ax_min_low_v = -4.5     # ax_min at zero velocity
        ax_min_high_v = -5.0    # ax_min at max velocity

        def ax_max_fun(vx):
            return ax_max_low_v + (ax_max_high_v - ax_max_low_v) * (vx / vx_points[-1])

        def ax_min_fun(vx):
            return ax_min_low_v + (ax_min_high_v - ax_min_low_v) * (vx / vx_points[-1])

        ax_max_grid = []
        ax_min_grid = []

        for ay in ay_points:
            for vx in vx_points:
                ax_max = ax_max_fun(vx)
                ax_min = ax_min_fun(vx)

                # Elliptical constraint: ax^2/(ax_max(vx)^2) + ay^2/(ay_max_val^2) <= 1
                factor = np.sqrt(max(0, 1 - (ay / ay_max_val) ** 2))
                ax_lim_max = ax_max * factor
                ax_lim_min = ax_min * factor

                ax_max_grid.append(ax_lim_max)
                ax_min_grid.append(ax_lim_min)

        self.spline_data = pygigi.GGVSplineData()
        self.spline_data.ay = ay_points.tolist()
        self.spline_data.vx = vx_points.tolist()
        self.spline_data.ax_max = ax_max_grid
        self.spline_data.ax_min = ax_min_grid
        self.spline_data.ay_max = [ay_max_val*0.99 for _ in vx_points]
        self.spline_data.ay_min = [-ay_max_val*0.99 for _ in vx_points]
        
        # GIGI model creation
        self.model_gigi = pygigi.FB_F1_10(self.spline_data)
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.05, self.control_loop)  #
        


    # -------------------------------
    # Callbacks
    # -------------------------------
    
    def imu_callback(self, msg):
        yaw_rate_deg = msg.imu.angular_velocity.z
        self.yaw_rate_measured = yaw_rate_deg * np.pi / 180.0  # rad/s
        
    def core_callback(self, msg):
        self.measured_speed = msg.state.speed / 4244.0      # m/s

    def initialpose_callback(self, msg):
        self.get_logger().info("Start index resetted")

    def pose_callback(self, msg):
        self.current_pose = msg.pose
        
        # Find closest point on the ClotoidList and update s_ego
        x_ego = self.current_pose.position.x
        y_ego = self.current_pose.position.y
        s_ego, _ = self.curve_list.findST1(x_ego, y_ego)
        
        self.s_ego = s_ego

        # Interpolation of target speed from raceline based on s_ego
        self.target_speed = np.interp(s_ego, self.raceline_s, self.raceline_vx)


    def control_loop(self):
        if self.current_pose is None:
            return

        ackermann_cmd = self.calculate_control_commands(self.s_ego)
        self.drive_pub.publish(ackermann_cmd)



    # === Core ===
    def calculate_control_commands(self, s_ego):
        ackermann_cmd = AckermannDriveStamped()

        if self.current_pose:
            x_ego = self.current_pose.position.x
            y_ego = self.current_pose.position.y
            quat = self.current_pose.orientation
            yaw_ego = self.euler_from_quaternion(quat.x, quat.y, quat.z, quat.w)[2]
            kappa_ego = self.yaw_rate_measured / max(self.measured_speed, 0.1)     # (rad/s)/(m/s) = 1/m

            # Dybamic lookahead point calculation based on current s_ego and dist_point_reached
            s_lookahead = s_ego + self.dist_point_reached
            x_lookahead, y_lookahead = self.curve_list.eval(s_lookahead)
            theta_lookahead = self.curve_list.theta(s_lookahead)
            kappa_lookahead = self.curve_list.theta_D(s_lookahead)

            # Marker
            self.target_marker.pose.position.x = x_lookahead
            self.target_marker.pose.position.y = y_lookahead
            self.target_marker_pub.publish(self.target_marker)

            # Pure Pursuit logic 
            quat = self.current_pose.orientation
            yaw = self.euler_from_quaternion(quat.x, quat.y, quat.z, quat.w)[2]

            dx = x_lookahead - self.current_pose.position.x
            dy = y_lookahead - self.current_pose.position.y
            L = np.hypot(dx, dy)

            local_x = dx * math.cos(-yaw) - dy * math.sin(-yaw)
            local_y = dx * math.sin(-yaw) + dy * math.cos(-yaw)

            if abs(local_y) > 1e-6:
                Radius = L**2 / (2 * abs(local_y))
                curv = 1.0 / Radius
                curv *= np.sign(local_y)
                steer = (self.wheel_base / Radius) *np.sign(local_y)
            
                theta = 2 * math.asin(L/(2 * Radius))
                arc_length = abs(Radius * theta)
            else:
                curv = 0.0
                arc_length = L     # if local_y is zero, consider straight line to the target
                steer = 0.0


            # Clothoid generation from the vehicle to the lookahead point as circular arc with curvature curv and length equal to the arc length to the lookahead point.
            self.curve.build(x_ego, y_ego, yaw_ego, curv, 0.0, arc_length)

            # === Sampling for MS-NN ===
            time_horizon = 0.05 * (self.num_points - 1) 
            vx_ls = max(self.target_speed, 0.01)
            length_to_sample = min(self.curve.length(), vx_ls * time_horizon)

            ds = 0.1
            N = max(int(length_to_sample / ds), self.num_points)
            s_list = np.linspace(0, length_to_sample, N, dtype=np.float64)

            x_list = np.zeros(s_list.size)
            y_list = np.zeros(s_list.size)
            curv_list = np.zeros(s_list.size)

            for i, s in enumerate(s_list):
                x_list[i], y_list[i] = self.curve.eval(s)
                curv_list[i] = self.curve.theta_D(s)

            vx_list = np.zeros(s_list.size)
            ax_list = np.zeros(s_list.size)
            t_list = np.zeros(s_list.size)

            for i in range(s_list.size):
                idx = np.argmin((self.raceline_x - x_list[i])**2 + (self.raceline_y - y_list[i])**2)
                vx_list[i] = self.raceline_vx[idx]
                ax_list[i] = self.raceline_ax[idx]
                if i > 0:
                    ds = np.hypot(x_list[i] - x_list[i-1], y_list[i] - y_list[i-1])
                    t_list[i] = t_list[i-1] + ds / max(vx_list[i], 0.01)


            # Update vx_init with the velocity from GIGI in the next iteration, to have a replanning of the GIGI profile at each iteration based on the current measured velocity of the vehicle. This allows to adapt the GIGI profile in real time to the actual conditions of the vehicle and to compensate for model mismatches and disturbances.
            self.vx_init = self.measured_speed   
            
            
            # === Initial velocity setting ===
            if self.measured_speed <= 0.5 and self.vx_init == 0.0:
                self.vx_init = 1.5 
                self.get_logger().info(f"[INIT] vx_init: {self.vx_init:.3f}")

            # Use GIGI replanner with updated velocity and timing
            if self.use_gigi:
                # Vectors of s and curvature for GIGI model
                SS = np.arange(0.0,arc_length, 0.1, dtype=np.float64)
                KK = [curv] * len(SS)
            
                # Calculation of replanning time based on the actual time elapsed since the last replanning, to have a more robust timing of the replanning and to avoid issues related to variable execution time of the control loop and of the GIGI optimization.
                now = self.get_clock().now()
                delta_t = (now - self.last_replan_time).nanoseconds * 1e-9  # [s]
                self.last_replan_time = now
            
                # Profile computation with GIGI using the current vx_init and the elapsed time since the last replanning to have a more adaptive and robust replanning of the velocity profile.
                total_time = self.model_gigi.compute(SS, KK, self.vx_init)

                # Temporale grid at 20Hz
                t_grid = [i * self.delta_t_s for i in range(self.num_points)]  # 15 points at 0.05s (20Hz)
                
                # Calculation of GIGI outputs (vx, ax, ay) at the time steps in t_grid to be used as inputs for the NN, to have a feedforward contribution from the GIGI optimization to the NN prediction and to allow the NN to learn to compensate for the delay in the GIGI profile and in the actuation of the vehicle.
                vx_list_GIGI = [self.model_gigi.evalV_t(t) for t in t_grid]
                ax_list_GIGI = [self.model_gigi.evalAx_t(t) for t in t_grid]
                ay_list_GIGI = [self.model_gigi.evalAy_t(t) for t in t_grid]
               

                # Velocity to send to the vehicle from the GIGI profile considering the estimated delay of the GIGI optimization and actuation pipeline, to have a more accurate compensation of the delay and to improve the performance of the controller.
                vx = self.model_gigi.evalV_t(2*self.gigi_delay)
                

                # Tensors for NN input
                vx_seq = torch.tensor(vx_list_GIGI, dtype=torch.float32).view(1, -1, 1)
                ax_seq = torch.tensor(ax_list_GIGI, dtype=torch.float32).view(1, -1, 1)
                ay_seq = torch.tensor(ay_list_GIGI, dtype=torch.float32).view(1, -1, 1)
                curv_seq = torch.full((1, vx_seq.shape[1], 1), fill_value=curv, dtype=torch.float32)   # Curvatura costante per il PP
                

         
            else:
                # Classic mode: PP + MS-NN (no velocity replanning from GIGI)

                # Interpolation for NN
                t_NN = np.linspace(0, 0.05 * (self.num_NN_points - 1), self.num_NN_points)
                vx_NN = np.interp(t_NN, t_list, vx_list)
                ax_NN = np.interp(t_NN, t_list, ax_list)
                
                # Sequences for NN
                vx_seq = torch.tensor(vx_NN, dtype=torch.float32).view(1, -1, 1)
                curv_seq = torch.full((1, vx_seq.shape[1], 1), fill_value=curv, dtype=torch.float32)   # Constant curvature for PP
                
                # Velocity for long. controller
                vx = vx_NN[0]


            
            # NN inference
            with torch.no_grad():
                output, _, _ = self.model.forward(curv_seq, vx_seq)
            angle_ff = float(output[0])

            # Updating dynamic lookahead point 
            self.calculate_point_reached_dist(angle_ff)
                
            # Commands
            ackermann_cmd.drive.steering_angle = np.clip(angle_ff, -self.max_steering_angle, self.max_steering_angle)
            ackermann_cmd.drive.speed = vx  
            ackermann_cmd.header.stamp = self.get_clock().now().to_msg()

        return ackermann_cmd


    # Function to calculate the distance to the lookahead point to consider the point reached based on the current velocity, the curvature of the path and the steering angle, to have a more adaptive and dynamic calculation of the lookahead point and to improve the performance of the controller in different conditions and phases of the track.
    def calculate_point_reached_dist(self, steering_angle):
        vx = max(self.target_speed, 0.01)
        time_horizon = 0.05 * (self.num_NN_points - 1)
        # Minimum required lookahead distance based on the current velocity and a fixed time horizon, to ensure that the lookahead point is far enough to allow the controller to react and to avoid issues related to too short lookahead distances at high speeds.
        min_required_lookahead = vx * time_horizon
        min_required_lookahead = np.clip(min_required_lookahead, self.min_dist_point_reached, self.max_dist_point_reached)

        # Average curvature between the closest point and the lookahead point to have a more accurate estimation of the curvature of the path in front of the vehicle and to improve the calculation of the reduction factor for the lookahead distance in case of high curvature.
        x_ego = self.current_pose.position.x
        y_ego = self.current_pose.position.y
        s_ego, _ = self.curve_list.findST1(x_ego, y_ego)
        kappa_closest = self.curve_list.theta_D(s_ego) 
        kappa_next = self.curve_list.theta_D(s_ego + self.dist_point_reached) 
        curvature_avg = (kappa_closest + kappa_next) / 2

        # Reduction factor based on curvature and steering angle to reduce the lookahead distance in case of high curvature and/or high steering angles, to improve the performance of the controller in these conditions and to avoid issues related to overshooting and instability.
        curvature_factor =  curvature_avg / self.kappa_max_road 
        steering_factor = abs(steering_angle) / self.max_steering_angle
        reduction_factor = min(steering_factor + curvature_factor, 1.0)

        # Final lookahead distance calculation considering the minimum required lookahead and the reduction factor, to have a more adaptive and dynamic lookahead distance that can improve the performance of the controller in different conditions and phases of the track.
        self.dist_point_reached = min_required_lookahead + 0*(1.0 - reduction_factor) * (self.max_dist_point_reached - min_required_lookahead)


    # Function to convert quaternion to euler angles, to extract the yaw angle from the vehicle pose and to use it in the control calculations.
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
    node = PPNNController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
