import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy, QoSProfile, HistoryPolicy

# import odometry
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan, Imu

import numpy as np 
import copy 
from matplotlib import rc
from dmp import dmp, obstacle_superquadric as obs

from matplotlib import pyplot as plt

#make cbf.py visible
import sys
sys.path.append('/home/daniele/colcon_ws/src/turtlebot3_dmp/turtlebot3_dmp')
from cbf import CBF

import math

import time

class Turtlebot3DMP(Node):

    def __init__(self):
        super().__init__('turtlebot3_dmp_node')

        # qos_profile = QoSProfile( history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE )

        # definition of publisher and subscriber object to /cmd_vel and /scan 
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.subscription = self.create_subscription(LaserScan, '/scan', self.laser_callback, rclpy.qos.qos_profile_sensor_data)
        # self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, rclpy.qos.qos_profile_sensor_data)

        self.dt = 0.1
        self.iter = 0



    def gen_dynamic_force(self, gamma, eta, v_ego, v_obs, p_obs, p_ego):
        """
        From the paper: "Duhé, J. F., Victor, S., & Melchior, P. (2021). Contributions on artificial potential field method for effective obstacle avoidance. Fractional Calculus and Applied Analysis, 24, 421-446.

        a_max: maximum acceleration
        eta: repulsive gain factor
        v_ego: velocity of the ego vehicle
        v_obs: velocity of the obstacle
        p_obs: position of the obstacle
        p_ego: position of the ego vehicle
        """
        v_rel = v_ego - v_obs
        p_rel = p_ego - p_obs
        rho_s = np.linalg.norm(p_rel)
        v_ro = np.dot(v_rel, -p_rel) / rho_s
        rho_m = v_ro**2 / (2 * gamma)
        rho_delta = rho_s - rho_m
        v_ro_orth = np.sqrt(np.linalg.norm(v_rel)**2 - v_ro**2)
        nabla_p = - eta * (1 + v_ro/gamma) / (rho_delta**2)
        nabla_v = eta * v_ro * v_ro_orth / (rho_delta**2) / (gamma * rho_s)
        return np.array([- nabla_p, - nabla_v])



    def gen_traj(self):
        K_approx = 0.0001  # approximation gain

        cbf = CBF()  # CBF initialization
        delta_0 = 0.05  # small constant for control barrier function
        eta = 0.25 # repulsive gain factor
        r_min = 0.25  # radius over which the repulsive potential field is active
        gamma = 100.0  # maximum acceleration for the robot
        # Reference trajectory (Cartesian coordinates)
        N = 1000  # discretization points
        a0 = 1.0  # ellipse major axis
        a1 = 1.0  # ellipse minor axis
        t = np.linspace(0,np.pi,N)  # time
        x = a0*np.sin(t)  # x
        y = a1*np.cos(t)  # y
        dx = a0*np.cos(t)  # dx
        dy = -a1*np.sin(t)  # dy
        ref_path = np.vstack((x,y)).T  # reference path
        ref_vel = np.vstack((dx,dy)).T  # reference velocity

        # DMPs training
        n_bfs = 100  # number of basis functions
        time_step = 0.01  # time-step (default 0.01)
        self.dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 115, dt = time_step, T = t[-1],
                                    alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs (K=115)
        self.dmp_traj.imitate_path(x_des = ref_path)  # train the DMPs



        
        obstacle_centers = []  # obstacle center list
        num_obstacles = 60  # number of obstacles
        radius = 0.15  # radius of the obstacles REAL
        for i in range(num_obstacles):
            obstacle_centers.append(np.array([-2.6, -0.2]) + radius * np.array([np.cos(2*np.pi*i/float(num_obstacles)), np.sin(2*np.pi*i/float(num_obstacles))]))  # obstacle center REAL
        




        # ALPHA = 50
        alpha = 50  # CBF gain
        obs_force = np.array([0.,0.])  # no obstacle external force

        self.dmp_traj.x_0 = np.array([-2., -0.5])  # new start in cartesian coordinates
        self.dmp_traj.x_goal = np.array([-2., 1.5])  # new goal in cartesian coordinates
        self.dmp_traj.reset_state()  # reset the state of the DMPs
        x_list = np.array(self.dmp_traj.x) # x, y
        x_dot_list = np.array(self.dmp_traj.dx)  # v_x, v_y
        x_ddot_list = np.array(self.dmp_traj.ddx)  # a_x, a_y
        print("STARTING DMP")
        while not np.linalg.norm(self.dmp_traj.x - self.dmp_traj.x_goal) < 0.01:
            external_force_total = np.array([0.,0.])
            for obstacle_center in obstacle_centers:
                external_force, _ = cbf.compute_u_safe_dmp_traj(self.dmp_traj, alpha = alpha, exp = 1.0, delta_0 = delta_0, eta = eta, r_min = r_min, gamma = gamma, obs_center = obstacle_center,
                                                            obs_force = obs_force, K_appr = K_approx, type = 'obstacle')
                external_force_total += external_force

            print(external_force_total)
            x, x_dot, x_ddot = self.dmp_traj.step(external_force = external_force_total)  # execute the DMPs
            x_list = np.vstack((x_list, x))
            x_dot_list = np.vstack((x_dot_list, x_dot))
            x_ddot_list = np.vstack((x_ddot_list, x_ddot))

            print("ITER " + str(len(x_list)))
            print("x_dot " + str(x_dot))

        # Save the learnt trajectory for the next part
        path_cbf = copy.deepcopy(x_list)
        vel_cbf = copy.deepcopy(x_dot_list)

    


        # Time step ratio
        undersample_factor = int(self.dt / time_step)

        # Undersample the paths and velocities
        path_cbf = self.undersample(path_cbf, undersample_factor)
        plt.plot(path_cbf[:, 0], path_cbf[:, 1], 'r', label='DMP path')
        plt.show()
        vel_cbf = self.undersample(vel_cbf, undersample_factor)

        self.v = np.linalg.norm(vel_cbf, axis=1)  # linear velocity
        theta = np.arctan2(vel_cbf[:, 1], vel_cbf[:, 0])  # initial angle
        theta = np.unwrap(theta)  # unwrap the angle
        self.omega = np.gradient(theta,self.dt) # angular velocity

        
    # UNDERSAMPLING
    def undersample(self, data, factor):
        return data[::factor]


    def control_loop(self):
        if self.iter < len(self.v):
            # self.publisher_.publish(Twist(linear=Vector3(x=self.v.pop(), y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=self.omega.pop())))
            print("PUBLISHING VELOCITY v " + str(self.v[self.iter]) + ", omega " + str(self.omega[self.iter]))
            self.publisher_.publish(Twist(linear=Vector3(x=self.v[self.iter], y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=self.omega[self.iter])))
        else:
            self.publisher_.publish(Twist(linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=0.0)))

        self.iter += 1



def main(args=None):
    rclpy.init(args=args)
    control_node = Turtlebot3DMP()
    control_node.gen_traj()
    control_node.create_timer(control_node.dt, control_node.control_loop)

    rclpy.spin(control_node)

    # control_node.get_logger().info('Publishing: "%s"' % msg.data)
    try:
        while rclpy.ok():
            pass
            # control_node.control_loop()
            # time.sleep(control_node.dt)
    except KeyboardInterrupt:
        pass

    control_node.destroy_node()
    rclpy.shutdown()










if __name__ == '__main__':
    main()
