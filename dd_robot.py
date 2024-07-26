## Differentia-Drive Robot python class

import numpy as np 
import scipy.linalg
import matplotlib.pyplot as plt
from dmp import derivative_matrices as dm

class PID(object):
    '''
    PID controller (one step update)

    Inputs:
    - Kp: proportional gain
    - Ki: integral gain
    - Kd: derivative gain
    - Ts: sampling time

    Outputs:
    - tau_1: control input 1
    - tau_2: control input 2
    - u0: PID output at previous instant of time
    - e1: error at k-1
    - e2: error at k-2
    '''
    def __init__(self, Kp=np.zeros(2), Ki=np.zeros(2), Kd=np.zeros(2), Ts=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ts = Ts

    def compute(self, error, u0, e1, e2, m, I):
        # Pid parameters
        error, u0, e1, e2, m, I = np.squeeze(error), np.squeeze(u0), np.squeeze(e1), np.squeeze(e2), np.squeeze(m), np.squeeze(I)  # to eleminate singleton dimensions

        alpha_1 = self.Kp[0] + self.Ki[0]*self.Ts + self.Kd[0]/self.Ts
        alpha_2 = self.Kp[1] + self.Ki[1]*self.Ts + self.Kd[1]/self.Ts
        beta_1 = self.Kp[0] + 2*self.Kd[0]/self.Ts
        beta_2 = self.Kp[1] + 2*self.Kd[1]/self.Ts
        gamma_1 = self.Kd[0]/self.Ts
        gamma_2 = self.Kd[1]/self.Ts

        # Discrete PID transfer function
        uv = u0[0] + alpha_1*error[0] - beta_1*e1[0] + gamma_1*e2[0]
        uw = u0[1] + alpha_2*error[1] - beta_2*e1[1] + gamma_2*e2[1]

        # Robot desired controls
        # tau_1 = m*uv/self.Ts
        # tau_2 = I*uw/self.Ts

        # Update PID parameters
        e2 = e1
        e1 = error
        u0 = np.array([uv, uw])

        return uv, uw, u0, e1, e2
    
class filters(object):
    '''
    Filters class: a class that contains different types of filters.
    '''
    def __init__(self, Ts = 0.01):
        self.Ts = Ts

    def moving_average(self, input_signal, window_size):
        '''
        Moving average filter.

        Inputs:
        - input_signal: input signal
        - window_size: window size

        Outputs:
        - output_signal: output signal
        '''
        filtered_signal_1 = []
        filtered_signal_2 = []
        input_signal_1 = input_signal[:,0]
        input_signal_2 = input_signal[:,1]  
        for i in range(len(input_signal_1)):
            if i < window_size:
                filtered_signal_1.append(np.mean(input_signal_1[0:i+1]))
                filtered_signal_2.append(np.mean(input_signal_2[0:i+1]))
            else:
                filtered_signal_1.append(np.mean(input_signal_1[i-window_size:i+1]))
                filtered_signal_2.append(np.mean(input_signal_2[i-window_size:i+1]))
        
        filtered_signal_1 = np.array(filtered_signal_1)
        filtered_signal_2 = np.array(filtered_signal_2)
        
        return filtered_signal_1, filtered_signal_2
        

    

class DD_robot(object):
    '''
    Differential drive robot class.
    
    Inputs:
    - X: system state (x, y, theta, v, w)
    - dt: time-step size
    - m: total mass
    - I: total moment of inertia
    - controller: PID controller
    '''
    def __init__(self, X=np.zeros(5), dt=0.01, m=0, I=0, controller=PID()):
        '''
        Constructor method.
        '''
        self.X = X  # system state (x, y, theta, v, w)
        self.dt = dt  # time-step size
        self.m = m  # total mass
        self.I = I  # total moment of inertia
        self.controller = controller

    def kinematic_model(self, v, w, sigma = None, B = None):
        '''
        Differential drive robot kinematic model.

        Inputs:
        - v: linear velocity
        - w: angular velocity
        - sigma: Gaussian process variance
        - B: Brownian motion increment

        Outputs:
        - X: system state (x, y, theta, v, w)
        '''
        if sigma is None:
            sigma = np.zeros(4)

        if B is None:
            B = np.zeros((4,1))

        self.X = np.reshape(self.X, (5,1))  # from array to column vector
        theta = self.X[2,0]  # orientation
        theta, v, w = np.squeeze(theta), np.squeeze(v), np.squeeze(w)  # to eleminate singleton dimensions
        fx = np.array([[v*np.cos(theta), v*np.sin(theta), w, 0, 0]]).transpose()  # drift vector
        gx = np.array([[sigma[0]*np.cos(theta), 0, 0, 0],[sigma[0]*np.sin(theta), 0, 0, 0],
                       [0, sigma[1], 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]])  # diffusion matrix
        self.X = self.X + self.dt*fx + np.dot(gx,B)  # state update (Euler - Maruyama)

        return self.X
        

    def dynamic_model(self, tau_1, tau_2, sigma, B:np.array):
        '''
        Differential drive robot dynamics model.

        Inputs:
        - tau_1: linear velocity input
        - tau_2: angular velocity input
        - sigma: Gaussian process variance
        - B: Brownian motion increment

        Outputs:
        - X: system state (x, y, theta, v, w)
        '''
        self.X = np.reshape(self.X, (5,1))  # from array to column vector
        theta = self.X[2,0]  # orientation
        v = self.X[3,0]  # linear velocity
        w = self.X[4,0]  # angular velocity
        fx = np.array([[v*np.cos(theta), v*np.sin(theta), w, 1/self.m*tau_1, 1/self.I*tau_2]]).transpose()  # drift vector
        gx = np.array([[sigma[0]*np.cos(theta), 0, 0, 0],[sigma[0]*np.sin(theta), 0, 0, 0],
                       [0, sigma[1], 0, 0],[0, 0, sigma[2]/self.m, 0],[0, 0, 0, sigma[3]/self.I]])  # diffusion matrix
        self.X = self.X + fx*self.dt + np.dot(gx,np.reshape(B,(4,1)))  # state update (Euler-Maruyama)

        return self.X

    def forward_dynamics(self):
        '''
        Differential drive robot forward dynamics. 
        '''
        theta = self.X[2]
        v = self.X[3]
        dot_x = v*np.cos(theta)
        dot_y = v*np.sin(theta)

        return dot_x, dot_y
    
    def inverse_dynamics(self, v, w):
        '''
        Differential drive robot inverse dynamics.
        '''
        tau_1 = self.m*v/self.dt  # driving force
        tau_2 = self.I*w/self.dt  # steering torque

        return tau_1, tau_2

    def inverse_kinematics(self, dot_x, dot_y):
        '''
        Differential drive robot inverse kinematics.
        '''
        theta = self.X[2]
        v = dot_x*np.cos(theta) + dot_y*np.sin(theta)
        w = self.X[4]

        return v, w
    
    def generate_train_path(self, ref_speed = 2.00, K = 2.00, waypoints = None, tol = 0.05, sigma = None):
        '''
        Generate training path, training velocity and training acceleration for the differential drive robot. 

        Inputs:
        - ref_speed: reference speed
        - K: proportional gain
        - waypoints: waypoints
        - tol: reaching tolerance
        - sigma: Gaussian process variance

        Outputs:
        - train_path: training path
        - train_vel: training velocity
        - train_acc: training acceleration
        '''
        # Parameters
        counter = 0  # waypoints counter 
        i = 0  # time counter
        i_dec = 0  # deceleration time counter
        alpha = 21  # deceleration factor
        rg = 3 * tol  # deceleration region radius

        # Loop initialization
        B = np.zeros((4,1))  # Brownian motion initial condition
        #max_iter = 1320  # maximum number of iterations
        robot_position = np.array([self.X[0], self.X[1]])  # robot current position
        goal_dist = np.linalg.norm(waypoints[-1,:] - robot_position.transpose())  # distance to goal
        tVec = []  # time vector
        train_path = []  # system state array
        tVec.append(i*self.dt)  # store time
        train_path.append(np.array([[self.X[0]],[self.X[1]]]).transpose())  # store state
        while goal_dist > tol: # and i < max_iter:
            wp_dist = np.linalg.norm(waypoints[counter,:] - robot_position.transpose())  # distance to waypoint
            if wp_dist < tol:
                counter = counter + 1
            
            if counter >= waypoints.shape[0]:  # check if all waypoints have been reached
                break

            # Robot Proportional control
            theta_d = np.arctan2(waypoints[counter,1] - robot_position[1], waypoints[counter,0] - robot_position[0])  # desired heading 
            error = theta_d - self.X[2]  # heading error
            w = K*error
            
            # If we are approaching the goal and we are sufficiently near, then the robot must slow down
            if counter == waypoints.shape[0]-1 and goal_dist <= rg: 
                v_in = ref_speed * np.exp(-alpha*i_dec*self.dt)  # deceleration
                w_in = w  # keep the same angular velocity
                sigma_in = np.zeros(4)  # no noise
                i_dec = i_dec + 1  # deceleration time counter
            else:
                # keep the robot at a constant reference speed
                v_in = ref_speed
                w_in = w
                sigma_in = sigma
            
            # Kinematics
            v_in, theta_d, w_in = np.squeeze(v_in), np.squeeze(theta_d), np.squeeze(w_in)  # to eleminate singleton dimensions
            B = B + np.sqrt(self.dt)*np.random.randn(4,1)  # Brownian motion increment
            self.X = self.kinematic_model(v_in, w_in, sigma_in, B)  # system state update

            # Updates
            i = i + 1  # time counter
            robot_position = np.array([self.X[0], self.X[1]])  # update robot position
            goal_dist = np.linalg.norm(waypoints[-1,:] - robot_position.transpose())  # update distance to goal
            tVec.append(i*self.dt)  # append time
            train_path.append(np.reshape(robot_position,(2,1)).transpose())  # append training path

        # Time vector
        tVec = np.array(tVec)

        # Reference trajectory
        train_path = np.array(train_path)
        train_path = np.squeeze(train_path)

        # Reference velocity
        D1 = dm.compute_D1(train_path.shape[0], self.dt)
        train_vel = np.dot(D1, train_path)

        # Reference acceleration
        D2 = dm.compute_D2(train_path.shape[0], self.dt)
        train_acc = np.dot(D2, train_path)

        return tVec, train_path, train_vel, train_acc


    #def simulate(self, error, u0, e1, e2):
    #    self.controller.compute(error, u0, e1, e2, self.m, self.I)
    