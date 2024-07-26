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
multi_acc = []
tol = 0.05  # reaching tolerance 
waypoints = np.array([[-6, 3], [2, 3], [2, -1], [3, -4]])  
for j in range(N):
    robot.X = X0  # reset initial state
    tVec, train_path, train_vel, train_acc = robot.generate_train_path(v_target, K, waypoints, tol, sigma = diff_coeff)  # train path
    multi_tVec.append(tVec)
    multi_path.append(train_path)
    multi_vel.append(train_vel)
    multi_acc.append(train_acc)

#filt_acc_1, filt_acc_2 = filter.moving_average(multi_acc[0], window_size = 10)

# Plotting
plt.figure()
for j in range(N):
    plt.plot(multi_path[j][:,0], multi_path[j][:,1],'b--')  # train path
plt.plot(waypoints[:,0], waypoints[:,1], 'ro')  # waypoints
plt.plot(X0[0], X0[1], 'bo')  # initial position
plt.legend(['multiple paths'])
plt.axis('equal')

plt.figure()
for j in range(N):
    plt.plot(multi_tVec[j], multi_vel[j][:,0],'b-')  # train path
plt.show()

print(">> End of script")