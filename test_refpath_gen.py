# Test script 1: test the reference path generation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from dd_robot import PID, DD_robot, filters
from dmp import dmp_cartesian as dmp
import copy

# CONTROLLER SETUP
K = 12  # proportional gain, for robot angular velocity
v_target = 2.00  # target linear velocity

# ROBOT SETUP
mass = 20  # total mass
l = 0.760  # robot length
b = 0.330  # robot semi-axle
Inertia = mass*(4*pow(b,2)+pow(l,2))/12  # total moment of inertia
X0 = np.array([0, -1, 2*np.pi/3, 0, 0])  # initial state (origin as initial position, pi/4 as initial orientation, robot at rest)
Ts = 0.01 # sampling time
diff_coeff = np.array([0.0015, 0.0015, 0.0015, 0.0015])  # Gaussian process variance
controller = PID()  # default PID controller (in this script we are not using it)
robot = DD_robot(X = X0, dt = Ts, m = mass, I = Inertia, controller = controller) # robot initialization

# WAYPOINTS GENERATION
tol = 0.05  # reaching tolerance 
waypoints = np.array([[-6, 3], [2, 3], [2, -1], [3, -4]])  # waypoints to follow ([[-6, 3], [2, 3], [2, -2], [0, 0]])
tVec, train_path, train_vel, train_acc = robot.generate_train_path(v_target, K, waypoints, tol, sigma = diff_coeff)  # train path

# FILTERING TRAINING PROFILES
win_size = 5  # window size
train_filpath = filters.moving_average(input_signal = train_path, window_size = win_size)
train_filvel = filters.moving_average(input_signal = train_vel, window_size = win_size)   
train_filacc = filters.moving_average(input_signal = train_acc, window_size = win_size)

# LEARNING
# In the initilaziation we must pass the final time of the training path as input, otherwise the DMP will 
# learn the path with T = 1.0 (default value). When you call rollout method, force tau = 1. 
MP_new = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = 250, K = 5000, dt = Ts, T = tVec[-1], alpha_s = 2.0, tol = 3.0 / 100, rescale = None, basis = "gaussian")
MP_new.imitate_path(x_des = copy.deepcopy(train_filpath), dx_des = copy.deepcopy(train_filvel), ddx_des = copy.deepcopy(train_filacc), t_des = copy.deepcopy(tVec))

# EXECUTION
MP_new.x_goal = train_filpath[-1]
MP_new.x_0 = train_filpath[0]
MP_new.reset_state()  # to reset the internal state of the DMP
learnt_traj_full = MP_new.rollout(tau = 1)  
learnt_traj = learnt_traj_full[0]
learnt_vel = learnt_traj_full[1]

plt.figure()
plt.plot(train_filpath[:,0], train_filpath[:,1],'b-')  # train path
plt.plot(learnt_traj[:,0], learnt_traj[:,1],'r--')  # DMP path
plt.plot(X0[0], X0[1], 'bo')  # initial position
plt.plot(waypoints[:,0], waypoints[:,1], 'ro')  # waypoints
plt.axis('equal')

plt.figure()
plt.subplot(2,1,1)
plt.plot(train_filvel[:,0],'b-')
plt.plot(learnt_vel[:,0],'r--')
plt.xlabel('Time [s]')
plt.ylabel('dx[m/s]')

plt.subplot(2,1,2)
plt.plot(train_filvel[:,1],'b-')
plt.plot(learnt_vel[:,1],'r--')
plt.xlabel('Time [s]')
plt.ylabel('dy[m/s]')

plt.figure()
plt.subplot(2,1,1)
plt.plot(train_filacc[:,0],'b-')
plt.plot(learnt_traj_full[2][:,0],'r--')
plt.xlabel('Time [s]')
plt.ylabel('ddx[m/s^2]')

plt.subplot(2,1,2)
plt.plot(train_filacc[:,1],'b-')
plt.plot(learnt_traj_full[2][:,1],'r--')
plt.xlabel('Time [s]')
plt.ylabel('ddy[m/s^2]')
plt.show()

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(tVec,train_acc[:,0],'b-')
# plt.xlabel('Time [s]')
# plt.ylabel('ddx[m/s^2]')

# plt.subplot(2,1,2)
# plt.plot(tVec,train_acc[:,1],'b-')
# plt.xlabel('Time [s]')
# plt.ylabel('ddy[m/s^2]')
# plt.show()

print(">> End of script")