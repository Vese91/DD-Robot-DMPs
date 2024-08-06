# Test script 3: test Interpolating Bezier curves as reference trajectory

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from dd_robot import PID, DD_robot, filters
from dmp import dmp_cartesian as dmp
import bezier_interp as bz
import copy

# REFERENCE PATH GENERATION
tol = 0.05  # reaching tolerance for waypoints
control_points = np.array([[2,1],[3,4],[5,7],[8,8],[11,7],[13,10],[7,13]])  # control points
m = 5  # number of points between control points
path,_ = bz.evaluate_bezier(control_points, m)  # evaluate Interpolating Bezier curves

# ROBOT INITIALIZATION
K = 15  # proportional gain, for robot angular velocity
v_target = 2.00  # target linear velocity
mass = 20  # total mass
l = 0.760  # robot length
b = 0.330  # robot semi-axle
Inertia = mass*(4*pow(b,2)+pow(l,2))/12  # total moment of inertia
X0 = np.array([path[0,0], path[0,1], np.arctan2(path[1,1] - path[0,1],path[1,0]-path[0,0]), 0, 0])  # initial state 
Ts = 0.01 # sampling time
diff_coeff = np.array([0.0015, 0.0015, 0.0015, 0.0015])  # Gaussian process variance
controller = PID()  # default PID controller (in this script we are not using it)
robot = DD_robot(X=X0, dt=Ts, m=mass, I=Inertia, controller=controller) # robot initialization

# TRAINING PROFILES 
tVec, train_path, train_vel = robot.generate_train_profiles_pp(v_target, K, path, tol, sigma=diff_coeff)  # train path



print("End of script")