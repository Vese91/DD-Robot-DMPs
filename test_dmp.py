import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from dd_robot import PID, DD_robot, filters
from dmp import dmp
import math
import bezier_interp as bz
import copy
from cbf import CBF

#dynamic parameters
mu_s = 0.1
# m = 3.9 #turtlebot 4
g = 9.81 # gravity acceleration [m/s^2]
alpha = 10 # extended class-K function parameter (straight line)
exp = 1 # exponent of the extended class-K function, it must be an odd number (leave it as 1)  

# Generate a ref trajectory (not a closed one!) 
K = 1000
alpha = 4.0
dt = 0.001
N = 1000
freq = 1/N/dt/3
t = np.linspace(0, N)*dt
x = np.cos(2*math.pi*freq*t)
y = 5*np.sin(2*math.pi*freq*t)

# Choose the path representation
#path = np.vstack((x, y)).T  # x, y
path = np.vstack((np.sqrt(x**2 + y**2), np.arctan2(y, x))).T  # rho, theta

#plt.plot(path[:,0], path[:,1], label='Reference trajectory')
#plt.show()

plt.plot(path[:,0]*np.cos(path[:,1]), path[:,0]*np.sin(path[:,1]), label='Reference trajectory')
plt.show()

# train and execute a dmp in polar coordinates 
n_bfs = 100
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = 250, K = 1000, dt = dt,
                            T = t[-1], alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")
# dmp_traj.w = np.zeros((2, n_bfs+1))
dmp_traj.imitate_path(x_des=path, t_des=t)

# Change the start and goal position
#dmp_traj.x_0 = np.array([0.5, 0])  # new start in cartesian coordinates
#dmp_traj.x_goal = np.array([-0.5, 0.2])  # new goal in cartesian coordinates

dmp_traj.x_0 = np.array([np.sqrt(0.5**2 + 0**2), np.arctan2(0, 0.5)])  # new start in polar coordinates
dmp_traj.x_goal = np.array([np.sqrt((-0.5)**2 + 0.2**2), np.arctan2(0.2, (-0.5))])  # new goal in polar coordinates

#no cbf
dmp_traj.reset_state()
x_list = np.array(dmp_traj.x) # rho, theta
x_dot_list = np.array(dmp_traj.x) # v_r, omega
x_ddot_list = np.array(dmp_traj.x)  
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < 0.01:
    x, x_dot, x_ddot = dmp_traj.step()
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# x_list, x_dot_list, _, _  = dmp_traj.rollout(tau = 1)  

plt.subplot(2,1,1)
plt.title('No CBF')
plt.plot(x_list[:,0]*np.cos(x_list[:,1]), x_list[:,0]*np.sin(x_list[:,1]), label='DMP without CBF')
plt.ylabel('y')
plt.xlabel('x')
plt.scatter(dmp_traj.x_goal[0]*np.cos(dmp_traj.x_goal[1]), dmp_traj.x_goal[0]*np.sin(dmp_traj.x_goal[1]), label='Goal', color='red')
plt.scatter(dmp_traj.x_0[0]*np.cos(dmp_traj.x_0[1]), dmp_traj.x_0[0]*np.sin(dmp_traj.x_0[1]), label='Start', color='green')
plt.legend()
plt.subplot(2,1,2)
plt.plot(x_list[:,0]*np.power(x_dot_list[:,1],2), label='rho * omega^2 without CBF')
plt.ylabel('rho * omega^2')
plt.xlabel('time')
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g')
plt.legend()

# plt.subplot(2,1,1)
# plt.title('No CBF')
# plt.plot(x_list[:,0], x_list[:,1], label='DMP without CBF')
# plt.ylabel('y')
# plt.xlabel('x')
# plt.scatter(dmp_traj.x_goal[0], dmp_traj.x_goal[1], label='Goal', color='red')
# plt.scatter(dmp_traj.x_0[0], dmp_traj.x_0[1], label='Start', color='green')
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(x_list[:,0]*np.power(x_dot_list[:,1],2), label='rho * omega^2 without CBF')
# plt.ylabel('rho * omega^2')
# plt.xlabel('time')
# plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g')
# plt.legend()

plt.show()

#with cbf
dmp_traj.reset_state()
x_list = np.array(dmp_traj.x)
x_dot_list = np.array(dmp_traj.x)
x_ddot_list = np.array(dmp_traj.x)
status_list = []
violated_constraint = []
cbf = CBF()
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < 0.01: 
    external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha, mu_s, g, exp)
    status_list.append(psi)
    x, x_dot, x_ddot = dmp_traj.step(external_force=external_force)
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

plt.subplot(2,1,1)
plt.title('With CBF')
plt.plot(x_list[:,0]*np.cos(x_list[:,1]), x_list[:,0]*np.sin(x_list[:,1]), label='DMP with CBF')
plt.scatter(dmp_traj.x_goal[0], dmp_traj.x_goal[1], label='Goal', color='red')
plt.scatter(dmp_traj.x_0[0]*np.cos(dmp_traj.x_0[1]), dmp_traj.x_0[0]*np.sin(dmp_traj.x_0[1]), label='Start', color='green')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.subplot(2,1,2)
plt.plot(x_list[:,0]*np.power(x_dot_list[:,1],2), label='rho * omega^2 with CBF')
plt.axhline(y = mu_s*g, color='r', linestyle='-', label='mu_s * g')
plt.ylabel('rho * omega^2')
plt.xlabel('time')
plt.legend()
plt.grid()

plt.show()

plt.subplot(3,1,1)
plt.plot(x_dot_list[:,0], label='v_r')
plt.plot(x_dot_list[:,1], label='omega')
plt.plot(x_list[:,0], label='rho')
plt.legend()
plt.grid()

plt.subplot(3,1,2)
plt.plot(x_list[:,0]*np.cos(x_list[:,1]), label='x')
plt.plot(x_list[:,0]*np.sin(x_list[:,1]), label='y')
plt.legend()
plt.grid()

plt.subplot(3,1,3)
plt.plot(status_list, label='psi')
plt.xlabel('time')
plt.legend()
plt.grid()

plt.show()

print('>> End of the script')



