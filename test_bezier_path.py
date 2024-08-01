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
m = 10  # number of points between control points
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
diff_coeff = 0*np.array([0.0015, 0.0015, 0.0015, 0.0015])  # Gaussian process variance
controller = PID()  # default PID controller (in this script we are not using it)
robot = DD_robot(X=X0, dt=Ts, m=mass, I=Inertia, controller=controller) # robot initialization

# TRAINING PROFILES 





plt.figure()
plt.plot(path[:,0],path[:,1],'b.-')
plt.plot(control_points[:,0],control_points[:,1],'ro')
plt.show()

print("End of script")