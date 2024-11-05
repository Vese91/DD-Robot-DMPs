import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from dd_robot import PID, DD_robot, filters
from dmp import dmp
import bezier_interp as bz
import copy

#generate a ref circular trajectory in polar coordinates (NX2)
N = 100
t = np.linspace(0, np.pi, N)
r = np.linspace(0, 1, N)
path = np.vstack((r, t)).T

# plt.plot(path[:,0]*np.cos(path[:,1]), path[:,0]*np.sin(path[:,1]), label='Reference trajectory')
# plt.show()

# train and execute a dmp in polar coordinates
n_bfs = 100
dmp_traj = dmp.DMPs_cartesian(n_dmps=2, n_bfs=n_bfs, dt=0.01, tol=0.01)
# dmp_traj.w = np.zeros((2, n_bfs+1))
dmp_traj.imitate_path(path)
dmp_traj.x_0 = np.array([0, 0])
dmp_traj.x_goal = np.array([2, np.pi])
dmp_traj.reset_state()

x_list = np.array(dmp_traj.x)
x_dot_list = np.array(dmp_traj.x)
x_ddot_list = np.array(dmp_traj.x)
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < 0.01:
    x, x_dot, x_ddot = dmp_traj.step()
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

plt.plot(x_list[:,0]*np.cos(x_list[:,1]), x_list[:,0]*np.sin(x_list[:,1]), label='DMP trajectory')
plt.show()





