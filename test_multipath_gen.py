# Test script 2: test the reference path generation with multiple paths

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from dd_robot import PID, filters, DD_robot

# PID controller
K = 12  # proportional gain, for robot angular velocity
v_target = 2.00  # target linear velocity

# Robot setup
mass = 20  # total mass
l = 0.760  # robot length
b = 0.330  # robot semi-axle
Inertia = mass*(4*pow(b,2)+pow(l,2))/12  # total moment of inertia
X0 = np.array([0, -1, 2*np.pi/3, 0, 0])  # initial state (origin as initial position, pi/4 as initial orientation, robot at rest)
Ts = 0.01 # sampling time
diff_coeff = np.array([0.0015, 0.0015, 0.0015, 0.0015])  # Gaussian process variance
controller = PID()  # default PID controller (in this script we are not using it)
filter = filters()  # filters
robot = DD_robot(X=X0, dt=Ts, m=mass, I=Inertia, controller=controller) # robot initialization

# Multiple training paths
N = 10  # number of paths
multi_tVec = []
multi_path = []
multi_vel = []
tol = 0.05  # reaching tolerance 
waypoints = np.array([[-6, 3], [2, 3], [2, -1], [3, -4]])  
for j in range(N):
    robot.X = X0  # reset initial state
    tVec, train_path, train_vel = robot.generate_train_profiles(v_target, K, waypoints, tol, sigma = diff_coeff)  # train path
    multi_tVec.append(tVec)
    multi_path.append(train_path)
    multi_vel.append(train_vel)


print(">> End of script")