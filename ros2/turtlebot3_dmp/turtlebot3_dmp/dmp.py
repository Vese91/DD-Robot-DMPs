import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy, QoSProfile, HistoryPolicy

# import odometry
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu

import numpy as np 
import copy 
from matplotlib import rc
from dmp import dmp, obstacle_superquadric as obs

from matplotlib import pyplot as plt

#make cbf.py visible
import sys
sys.path.append('/home/daniele/colcon_ws/src/turtlebot3_follow/turtlebot3_follow')
from cbf import CBF

import math

import time
import threading

class Turtlebot3DMP(Node):

    def __init__(self):
        super().__init__('turtlebot3_dmp_node')

        qos_profile = QoSProfile( history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE )

        # definition of publisher and subscriber object to /cmd_vel and /scan 
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', qos_profile=qos_profile)
        # self.subscription = self.create_subscription(LaserScan, '/scan', self.laser_callback, rclpy.qos.qos_profile_sensor_data)
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, rclpy.qos.qos_profile_sensor_data)

        self.angle_min = 0
        self.angle_increment = 0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.ranges = []
        # list of distance in angles range
        self.range_view = []
        # list of rays angles in angles range
        self.angle_view = []
        self.stop = False  # seconds
        self.robot_pose = None

        self.dt = 0.05
        self.mu_s = 0.7
        self.g = 9.81
        self.cbfs = True
        self.obst = False

        self.init_task()
        self.train_dmps()





    def set_dmp_goal(self):
        while self.robot_pose is None:
            pass
        
        self.goal_pose = copy.deepcopy(self.robot_pose[:2])  # goal in cartesian coordinates
        self.goal_pose[0] += .5  # move the goal
        self.goal_pose[1] += 1.  # move the goal
        self.dmp_traj.x_0 = self.robot_pose[:2]  # new start in cartesian coordinates
        self.dmp_traj.x_goal = self.goal_pose  # new goal in cartesian coordinates
        self.dmp_traj.reset_state()
        #print x_0 and x_goal
        self.get_logger().info('Goal: %s' % self.dmp_traj.x_goal)
        self.get_logger().info('Start: %s' % self.dmp_traj.x_0)

        self.x_track = np.array(self.dmp_traj.x) # x, y
        self.dx_track = np.array(self.dmp_traj.dx)  # v_x, v_y
        self.ddx_track = np.array(self.dmp_traj.ddx)  # a_x, a_y
        self.theta_track = np.array([0])
    

    def euler_from_quaternion(self, orientation):
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w
        # roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)    
        # yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw
    

    def odom_callback(self, msg):
        # get the position and orientation of the robot
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # get the orientation of the robot
        orientation = msg.pose.pose.orientation
        # get the orientation in euler angles
        roll, pitch, yaw = self.euler_from_quaternion(orientation)
        # get the linear and angular velocity of the robot
        # linear_vel = msg.twist.twist.linear.x
        # angular_vel = msg.twist.twist.angular.z
        # get the position and orientation of the robot
        self.robot_pose = np.array([x, y, yaw])
        # get the linear and angular velocity of the robot
        # self.robot_vel = np.array([linear_vel, angular_vel])

    
    def rollout(self):
        # ROLLOUT
        self.x_track, self.dx_track, self.ddx_track, _ = self.dmp_traj.rollout()
        self.vx, self.omega = self.get_ddmr_velocities(self.x_track, self.dx_track, self.dt)



    def train_dmps(self):
        # REFERENCE TRAJECTORY (Cartesian coordinates) 
        T = 10. # period
        N = 1000  # discretization points
        a0 = 1  # ellipse major axis
        a1 = 0.5  # ellipse minor axis
        t = np.linspace(0,T/2.,N)  # time
        x = a0*np.cos(2*np.pi*t/T)  # x
        y = a1*np.sin(2*np.pi*t/T)  # y
        dx = -a0*2*np.pi/T*np.sin(2*np.pi*t/T)  # dx
        dy = a1*2*np.pi/T*np.cos(2*np.pi*t/T)  # dy
        # x = t
        # y = t
        # dx = np.ones(len(t))
        # dy = np.ones(len(t))
        ref_path = np.vstack((x,y)).T  # reference path
        ref_vel = np.vstack((dx,dy)).T  # reference velocity

        # plt.plot(ref_path[:,0], ref_path[:,1],'b-',label='Reference trajectory')
        # plt.plot(ref_path[0,0], ref_path[0,1],'ko',label='Start')
        # plt.plot(ref_path[-1,0], ref_path[-1,1],'kx',label='Goal')
        # plt.title('Ref. training trajectory')
        # plt.legend()
        # plt.show()

        # DMPs TRAINING
        self.dmp_traj.reset_state()  # reset the state of the DMPs
        self.dmp_traj.imitate_path(x_des=ref_path)  # train the DMPs
        self.publisher_.publish(Twist())
        # #plot dmp after rollout
        # x, _, _, _ = self.dmp_traj.rollout()
        # plt.plot(x[:,0], x[:,1])
        # plt.show()
    


    def init_task(self):
        # INIT DMPs
        n_bfs = 100  # number of basis functions
        self.dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 115, dt = self.dt, T=10.,
                                    alpha_s = 2.0, tol = 1.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs
        self.iter = 0

        self.obstacles = []
        self.obst_centers = []
        if self.obst:
            # OBSTACLES
            obstacle_center = np.array([0., -0.4])  # obstacle center
            self.obst_centers.append(obstacle_center)
            radius = 0.2
            obstacle_axis = np.ones(self.dmp_traj.n_dmps) * radius
            # superquadric parameters
            lmbda = 5.0
            beta = 5.0
            eta = 5.0
            self.obstacles.append(obs.Obstacle_Dynamic(center = obstacle_center, axis = obstacle_axis, 
                                    lmbda = lmbda, beta=beta, eta=eta, coeffs = np.ones(self.dmp_traj.n_dmps)))
        
        if self.cbfs:
            # CBF PARAMETERS
            self.alpha = 200 # extended class-K function parameter (straight line)
            self.exp = 1 # exponent of the extended class-K function, it must be an odd number (leave it as 1)
            self.cbf = CBF()
    

    def get_ddmr_velocities(self, traj_pos, traj_vel, dt):
        # Calculate the tangential orientation and angular velocity
        orient = np.arctan2(np.gradient(traj_pos[:, 1]), np.gradient(traj_pos[:, 0]))  # orientation angle
        orient = np.unwrap(orient)  # unwrap the orientation angle (to avoid jumps)
        angvel = np.gradient(orient) / dt  # Calculate the angular velocity

        vref = []  # reference velocity list
        for i in range(len(orient)):
            theta = orient[i]  # current orientation
            A = np.array([[np.cos(theta),np.sin(theta),0],[0,0,1]])  # inverse kinematics matrix
            b = np.array([traj_vel[i,0], traj_vel[i,1], angvel[i]])  # velocity vector in inertial frame
            vref_i = np.matmul(A,b)  # reference velocity
            vref.append(vref_i)  # reference velocity
        
        # Convert the list to a numpy array
        vref = np.array(vref)  # reference velocity (shape (N,2))
        vx_ref = vref[:,0]  # reference forward velocity
        omega_ref = vref[:,1]  # reference angular velocity

        return vx_ref, omega_ref
        
            
    def control_loop(self):
        # if self.iter == 0:
        self.init_time = time.time()
        # else:
        #     self.get_logger().info('Time: %f' % (time.time() - self.init_time))
        #     self.init_time = time.time()
        # PLANNING LOOP
        if np.linalg.norm(self.dmp_traj.x - self.dmp_traj.x_goal) > self.dmp_traj.tol:
            external_force = np.zeros(self.dmp_traj.n_dmps)
            for obstacle in self.obstacles:
                curr_obst = copy.deepcopy(obstacle)
                freq_obst = 2
                amp_obst = 0.0
                vel_obst = np.array([0, amp_obst * 2*math.pi*2*np.cos(2*math.pi*freq_obst*self.iter*self.dmp_traj.cs.dt)])  # velocity of the curr_obst
                external_force += curr_obst.gen_external_force(self.dmp_traj.x, self.dmp_traj.dx - vel_obst)
                curr_obst.center += np.array([0, amp_obst * np.sin(2*math.pi*freq_obst*self.iter*self.dmp_traj.cs.dt)])  # move the curr_obst
                self.obst_centers.append(curr_obst.center)
            if self.cbfs:
                cbf_force, _ = self.cbf.compute_u_safe_dmp_traj(self.dmp_traj, self.alpha, self.mu_s, self.g, self.exp, external_force)
                external_force += cbf_force
            x, x_dot, x_ddot = self.dmp_traj.step(external_force=external_force)  # execute the DMPs      
            self.x_track = np.vstack((self.x_track, x))
            self.dx_track = np.vstack((self.dx_track, x_dot))
            self.ddx_track = np.vstack((self.ddx_track, x_ddot))

            vx = np.linalg.norm(x_dot)
            if vx > 0:
                theta = np.arctan2(x_dot[1], x_dot[0])
            else:
                theta = self.theta_track[-1]
            self.theta_track = np.append(self.theta_track, theta)
            dtheta = theta - self.theta_track[-2]
            omega = dtheta / self.dt

            # vx, omega = self.get_ddmr_velocities(self.x_track, self.dx_track, self.dt)

            msg_pub = Twist()
            msg_pub.linear.x = vx
            msg_pub.angular.z = omega
            self.publisher_.publish(msg_pub)
            self.iter += 1

        else:
            self.stop_robot()
            self.plot_traj()
            self.destroy_node()
            rclpy.shutdown()
        
        if time.time() - self.init_time < self.dt:
            time.sleep(self.dt - (time.time() - self.init_time))
        
        # self.get_logger().info('Time: %f' % (time.time() - init_time))




    def stop_robot(self):
        self.stop = True
        msg_pub = Twist()
        self.publisher_.publish(msg_pub)


    def plot_traj(self):
        # PLOT TRAJECTORY
        plt.plot(self.x_track[:,0], self.x_track[:,1],'b-',label='DMPs trajectory')
        plt.plot(self.x_track[0,0], self.x_track[0,1],'ko',label='Start')
        plt.plot(self.x_track[-1,0], self.x_track[-1,1],'kx',label='Goal')
        # for obst_center in self.obst_centers:
        #     plt.plot(obst_center[0], obst_center[1], 'ro', label='Obstacle')
        plt.title('DMPs trajectory')
        plt.legend()
        plt.show()
        


def main(args=None):
    rclpy.init(args=args)

    turtlebot3_dmp_node = Turtlebot3DMP()

    t = threading.Thread(target=rclpy.spin, args=[turtlebot3_dmp_node])
    t.start()

    turtlebot3_dmp_node.set_dmp_goal()
    # turtlebot3_dmp_node.rollout()
    turtlebot3_dmp_node.create_timer(turtlebot3_dmp_node.dt, turtlebot3_dmp_node.control_loop)

    try:
        while rclpy.ok():
            pass
    except KeyboardInterrupt:
        turtlebot3_dmp_node.stop_robot()
        # time.sleep(0.5)

    # Destroy the node explicitly

    turtlebot3_dmp_node.destroy_node()
    rclpy.shutdown()

    t.join()


if __name__ == '__main__':
    main()
