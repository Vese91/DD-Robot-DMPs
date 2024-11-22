# Test script 3: test Interpolating Bezier curves as reference trajectory

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from dd_robot import PID, DD_robot, filters
from dmp import dmp
import bezier_interp as bz
import copy
import math 

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
diff_coeff = np.array([0.0015, 0.0015, 0.0015, 0.0015])  # Gaussian process variance
controller = PID()  # default PID controller (in this script we are not using it)
robot = DD_robot(X=X0, dt=Ts, m=mass, I=Inertia, controller=controller) # robot initialization

# TRAINING PROFILES 
tVec, train_path, train_vel = robot.generate_train_profiles(v_target, K, path, tol, sigma=diff_coeff)  # train path


N = 1000
freq = 1/N/2
t = np.linspace(0, N)
x = np.cos(2*math.pi*freq*t)
y = 5*np.sin(2*math.pi*freq*t)
#x = copy.deepcopy(t) 
#y = 0.0001*(x - N/2)**2
train_path = np.vstack((x, y)).T  # x, y

# DMPs
MP_new = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = 250, K = 1000, dt = Ts,
                            T = tVec[-1], alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")
# MP_new.imitate_path(x_des = copy.deepcopy(train_path), dx_des = copy.deepcopy(train_vel), t_des = copy.deepcopy(tVec))
#MP_new.imitate_path(x_des = copy.deepcopy(train_path), t_des = copy.deepcopy(tVec))
MP_new.imitate_path(x_des = copy.deepcopy(train_path), t_des = t)

# EXECUTION
MP_new.x_0 = np.array([0.5, 0]) # train_path[-1]
MP_new.x_goal = np.array([-0.5, 0.2]) # train_path[0]
MP_new.reset_state()  # to reset the internal state of the DMP
learnt_traj_full = MP_new.rollout(tau = 1)  
learnt_traj = learnt_traj_full[0]
learnt_vel = learnt_traj_full[1]


plt.figure()
plt.subplot(3,1,1)
#plt.plot(path[:,0],path[:,1],'b.-')
#plt.plot(control_points[:,0],control_points[:,1],'go')
plt.legend(['ref path (Interp Bezier)','control points'])
plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.subplot(3,1,2)
plt.plot(train_path[:,0],train_path[:,1],'b--')
#plt.plot(control_points[0,0],control_points[0,1],'ro')
#plt.plot(control_points[-1,0],control_points[-1,1],'bo')
plt.legend(['train path (DD robot)','start','goal'])
plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.subplot(3,1,3)
plt.plot(learnt_traj[:,0],learnt_traj[:,1],'r--')
#plt.plot(control_points[0,0],control_points[0,1],'ro')
#plt.plot(control_points[-1,0],control_points[-1,1],'bo')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.show()

print("End of script")