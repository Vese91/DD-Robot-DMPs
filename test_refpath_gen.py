# Test script 1: test the reference path generation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from dd_robot import PID, DD_robot
from dmp import dmp_cartesian as dmp

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
robot = DD_robot(X=X0, dt=Ts, m=mass, I=Inertia, controller=controller) # robot initialization

# Waypoints
tol = 0.05  # reaching tolerance 
waypoints = np.array([[-6, 3], [2, 3], [2, -1], [3, -4]])  # waypoints to follow ([[-6, 3], [2, 3], [2, -2], [0, 0]])
tVec, train_path, train_vel, train_acc = robot.generate_train_path(v_target, K, waypoints, tol, sigma = diff_coeff)  # train path

# Learning
MP_new = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = 100, K = 1000, dt = Ts, alpha_s = 4.0, tol = 6.0 / 100, rescale=None)
MP_new.imitate_path(x_des = train_path, t_des = tVec)

# Execution
MP_new.x_goal = train_path[-1]
MP_new.x_0 = train_path[0]
MP_new.reset_state()  # to reset the internal state of the DMP
learnt_traj = MP_new.rollout()[0]
learnt_vel = MP_new.rollout()[1]

plt.figure()
plt.plot(train_path[:,0], train_path[:,1],'b-')  # train path
plt.plot(learnt_traj[:,0], learnt_traj[:,1],'r--')  # DMP path
plt.plot(X0[0], X0[1], 'bo')  # initial position
plt.plot(waypoints[:,0], waypoints[:,1], 'ro')  # waypoints
plt.axis('equal')


plt.figure()
plt.subplot(2,1,1)
plt.plot(train_vel[:,0],'b-')
plt.plot(learnt_vel[:,0],'r--')
plt.xlabel('Time [s]')
plt.ylabel('dx[m/s]')

plt.subplot(2,1,2)
plt.plot(train_vel[:,1],'b-')
plt.plot(learnt_vel[:,1],'r--')
plt.xlabel('Time [s]')
plt.ylabel('dy[m/s]')

plt.figure()
plt.subplot(2,1,1)
plt.plot(tVec,train_acc[:,0],'b-')
plt.xlabel('Time [s]')
plt.ylabel('ddx[m/s^2]')

plt.subplot(2,1,2)
plt.plot(tVec,train_acc[:,1],'b-')
plt.xlabel('Time [s]')
plt.ylabel('ddy[m/s^2]')
plt.show()

print(">> End of script")