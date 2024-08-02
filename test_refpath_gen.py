# Test script 1: test the reference path generation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from dd_robot import PID, DD_robot, filters
from dmp import dmp_cartesian as dmp
import copy

# CONTROLLER SETUP
K = 15  # proportional gain, for robot angular velocity
v_target = 2.00  # target linear velocity

# ROBOT SETUP
mass = 20  # total mass
l = 0.760  # robot length
b = 0.330  # robot semi-axle
Inertia = mass*(4*pow(b,2)+pow(l,2))/12  # total moment of inertia
X0 = np.array([9, -2, np.pi/2 +np.pi/6, 0, 0])  # initial state (origin as initial position, pi/4 as initial orientation, robot at rest)
Ts = 0.01 # sampling time
diff_coeff = np.array([0.0015, 0.0015, 0.0015, 0.0015])  # Gaussian process variance
controller = PID()  # default PID controller (in this script we are not using it)
robot = DD_robot(X = X0, dt = Ts, m = mass, I = Inertia, controller = controller) # robot initialization




print(">> End of script")