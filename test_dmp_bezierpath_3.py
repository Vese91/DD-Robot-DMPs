import numpy as np 
import copy 
import matplotlib.pyplot as plt
from matplotlib import rc
from dmp import dmp, obstacle_superquadric as obs
import bezier_interp as bz
from cbf import CBF

'''
CBFs = Control Barrier Functions
DMPs = Dynamic Movement Primitives

CBFs are used to ensure safety in the system, while DMPs are used to generate smooth trajectories.
'''

# Dynamic parameters
mu_s = 0.7  # static friction coefficient
g = 9.81 # gravity acceleration [m/s^2]
alpha = 25 # extended class-K function parameter (straight line)
exp = 1 # exponent of the extended class-K function, it must be an odd number (leave it as 1)

# Reference trajectory (Cartesian coordinates) 
# Bezier curve of n-degree
Tf = 2.0  # final time
Ts = 0.01  # sampling time
control_points = np.array([[1,1],[3,1],[5,1],[7,1],[7,-1],[5,-1],[3,-1],[1,-1]])  # example 1
theta = np.pi / 2  # rotation angle (pi/2 radians counter clockwise)
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
control_points = np.dot(control_points, rotation_matrix.T) # example 1 (rotated)
tVec, ref_path, ref_vel = bz.regular_bezier_curve(control_points, Tf, Ts)  # reference trajectory

# N = 1000  # discretization points
# a0 = 3.0  # ellipse major axis
# a1 = 1.0  # ellipse minor axis
# t = np.linspace(0,np.pi,N)  # time
# x = a0*np.cos(t)  # x
# y = a1*np.sin(t)  # y
# dx = -a0*np.sin(t)  # dx
# dy = a1*np.cos(t)  # dy
# ref_path = np.vstack((x,y)).T  # reference path
# ref_vel = np.vstack((dx,dy)).T  # reference velocity

# DMPs training
n_bfs = 100  # number of basis functions
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 200, dt = 0.01, T = Tf,
                              alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs
dmp_traj.imitate_path(x_des=ref_path)  # train the DMPs

# DMPs execution (no CBF)
dmp_traj.x_0 = control_points[0]  # new start in cartesian coordinates np.array([-3,6])
dmp_traj.x_goal = control_points[-1]  # new goal in cartesian coordinates np.array([4,-2])
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# Loop
goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    x, x_dot, x_ddot = dmp_traj.step()  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

plt.plot(ref_path[:,0], ref_path[:,1],'r--',label='Reference trajectory')
plt.plot(ref_path[0,0], ref_path[0,1],'ko',label='Start')
plt.plot(ref_path[-1,0], ref_path[-1,1],'kx',label='Goal')
plt.plot(x_list[:,0], x_list[:,1],'b-',label='DMPs trajectory')
plt.plot(x_list[0,0], x_list[0,1],'go',label='Start')
plt.plot(x_list[-1,0], x_list[-1,1],'rx',label='Goal')
plt.legend()
plt.show()

plt.subplot(2,1,1)
plt.plot(ref_vel[:,0],'r--',label='Ref. velocity (dx)')
plt.plot(x_dot_list[:,0],'b-',label='DMPs velocity (dx)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(ref_vel[:,1],'r--',label='Ref. velocity (dy)')
plt.plot(x_dot_list[:,1],'b-',label='DMPs velocity (dy)')
plt.legend()
plt.show()

# Centrifugal force
F_cf1 = (ref_path[:,0]*ref_vel[:,1]-ref_path[:,1]*ref_vel[:,0])**2/((ref_path[:,0]**2+ref_path[:,1]**2)**(3/2))  # centrifugal force in ref path
F_cf2 = ((x_list[:,0]*x_dot_list[:,1]-x_list[:,1]*x_dot_list[:,0])**2/((x_list[:,0]**2+x_list[:,1]**2)**(3/2)))  # centrifugal force in learnt path

plt.subplot(2,1,1)
plt.plot(F_cf1, 'b--',label='Centrifugal force (ref)')  # centrifugal force
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g') # static friction (centripeal force)
plt.legend()
plt.title('Centrifugal force (no CBF)')

plt.subplot(2,1,2)
plt.plot(F_cf2, 'b-',label='Centrifugal force (learnt)')  # centrifugal force
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g') # static friction (centripeal force)
plt.legend()
plt.show()

print(">> End of the script")