import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
from dmp import dmp
import bezier_interp as bz
from cbf import CBF

'''
CBFs = Control Barrier Functions
DMPs = Dynamic Movement Primitives

CBFs are used to ensure safety in the system, while DMPs are used to generate smooth trajectories.
'''

# Dynamic parameters
mu_s = 0.1  # static friction coefficient
g = 9.81 # gravity acceleration [m/s^2]
alpha = 10 # extended class-K function parameter (straight line)
exp = 1 # exponent of the extended class-K function, it must be an odd number (leave it as 1)

# Reference trajectory (Cartesian coordinates) 
#m = 20  # number of points between control points
#waypoints = np.array([[1.5,3],[3.5,2.92],[5,2.7],[6,2],[5,1.31],[3.49,1.11],[1.5,1.0]]) # control points 
#waypoints = np.array([[-2,2],[-0.53,1.71],[0.81,1],[-0.76,0.30],[-1.45,-0.70],[-0.43,-1.21],[1.26,-1.29],[3.00,-0.93]])
#ref_path, ref_vel = bz.evaluate_bezier(waypoints, m)  # evaluate Interpolating Bezier curves

# Reference trajectory (Polar coordinates) 
#ref_path_polar,_= bz.convert_to_polar_coord(ref_path, ref_vel) # convert Cartesian to Polar coordinates 

# Comparison between Cartesian and Polar coordinates
#plt.plot(ref_path[:,0], ref_path[:,1],'b-', label='Reference trajectory cartesian')
plt.plot(ref_path_polar[:,0]*np.cos(ref_path_polar[:,1]), ref_path_polar[:,0]*np.sin(ref_path_polar[:,1]),'b-', label='Reference trajectory polar')
plt.plot(waypoints[:,0], waypoints[:,1], 'ro', label='Control points')
plt.title('Reference trajectory for training')
plt.legend()
plt.show()

# DMPs training
n_bfs = 50  # number of basis functions
dmp_traj = dmp.DMPs_cartesian(n_dmps=2, n_bfs=n_bfs, dt=0.01, tol=0.01)  # initialize the DMPs
dmp_traj.imitate_path(ref_path_polar)  # generate the set of parameters to imitate reference path

# DMPs execution 
x0_new = np.array([1.5, 2.50])  # new initial state(cartesian coordinates)
goal_new = np.array([1.5,1.0])  # new goal state (cartesian coordinates)
x0_new_polar = np.array([np.sqrt(x0_new[0]**2 + x0_new[1]**2), np.arctan2(x0_new[1], x0_new[0])]) # new initial state (polar coordinates)
goal_new_polar = np.array([np.sqrt(goal_new[0]**2 + goal_new[1]**2), np.arctan2(goal_new[1], goal_new[0])])  # new goal state (polar coordinates)
dmp_traj.x_0 = x0_new_polar  # set a new initial state
dmp_traj.x_goal = goal_new_polar  # set a new goal state
dmp_traj.reset_state()  # reset the state of the DMPs

# No CBF
x_list = np.array(dmp_traj.x) # rho, theta
x_dot_list = np.array(dmp_traj.x) # v_r, omega
x_ddot_list = np.array(dmp_traj.x) # a_r, omega_dot
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < 0.01: # loop until the goal is reached
    x, x_dot, x_ddot = dmp_traj.step()  # call the DMPs step function
    x_list = np.vstack((x_list, x)) # store the states
    x_dot_list = np.vstack((x_dot_list, x_dot)) # store the velocities
    x_ddot_list = np.vstack((x_ddot_list, x_ddot)) # store the accelerations

# Plot the results (No CBF)
plt.subplot(2,1,1)
plt.plot(ref_path_polar[:,0]*np.cos(ref_path_polar[:,1]), ref_path_polar[:,0]*np.sin(ref_path_polar[:,1]),'b--', label='ref. training trajectory')
plt.plot(ref_path_polar[-1,0]*np.cos(ref_path_polar[-1,1]), ref_path_polar[-1,0]*np.sin(ref_path_polar[-1,1]),'rx',label='old goal')
plt.plot(ref_path_polar[0,0]*np.cos(ref_path_polar[0,1]), ref_path_polar[0,0]*np.sin(ref_path_polar[0,1]),'gx',label='old start')
plt.plot(x_list[:,0]*np.cos(x_list[:,1]), x_list[:,0]*np.sin(x_list[:,1]),'b-', label='DMP without CBF')
plt.plot(dmp_traj.x_goal[0]*np.cos(dmp_traj.x_goal[1]), dmp_traj.x_goal[0]*np.sin(dmp_traj.x_goal[1]),'ro',label='new goal')
plt.plot(dmp_traj.x_0[0]*np.cos(dmp_traj.x_0[1]), dmp_traj.x_0[0]*np.sin(dmp_traj.x_0[1]),'go',label='new start')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Plot centrifugal force (No CBF)
plt.subplot(2,1,2)
plt.axhline(y = mu_s*g, color='r', linestyle='-', label='mu_s * g')
plt.plot(x_list[:,0]*np.power(x_dot_list[:,1],2), label='rho * omega^2 without CBF')
plt.legend()
plt.show()

# With CBF
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x)  # rho, theta
x_dot_list = np.array(dmp_traj.x)  # v_r, omega
x_ddot_list = np.array(dmp_traj.x)  # a_r, omega_dot
status_list = []  # list to store the status of the CBF
cbf = CBF()  # initialize the CBF class
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < 0.01: # loop until the goal is reached
    external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha, mu_s, g, exp)  # compute the external force and the CBF status
    status_list.append(psi)  # store the status of the CBF
    x, x_dot, x_ddot = dmp_traj.step(external_force=external_force)  # call the DMPs step function
    x_list = np.vstack((x_list, x))  # store the states
    x_dot_list = np.vstack((x_dot_list, x_dot))  # store the velocities
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))  # store the accelerations

# Plot the results (With CBF)
plt.subplot(2,1,1)
plt.plot(ref_path_polar[:,0]*np.cos(ref_path_polar[:,1]), ref_path_polar[:,0]*np.sin(ref_path_polar[:,1]),'b--', label='ref. training trajectory')
plt.plot(ref_path_polar[-1,0]*np.cos(ref_path_polar[-1,1]), ref_path_polar[-1,0]*np.sin(ref_path_polar[-1,1]),'rx',label='old goal')
plt.plot(ref_path_polar[0,0]*np.cos(ref_path_polar[0,1]), ref_path_polar[0,0]*np.sin(ref_path_polar[0,1]),'gx',label='old start')
plt.plot(x_list[:,0]*np.cos(x_list[:,1]), x_list[:,0]*np.sin(x_list[:,1]),'b-', label='DMP with CBF')
plt.plot(dmp_traj.x_goal[0]*np.cos(dmp_traj.x_goal[1]), dmp_traj.x_goal[0]*np.sin(dmp_traj.x_goal[1]),'ro',label='new goal')
plt.plot(dmp_traj.x_0[0]*np.cos(dmp_traj.x_0[1]), dmp_traj.x_0[0]*np.sin(dmp_traj.x_0[1]),'go',label='new start')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Plot centrifugal force (CBF)
plt.subplot(2,1,2)
plt.axhline(y = mu_s*g, color='r', linestyle='-', label='mu_s * g')
plt.plot(x_list[:,0]*np.power(x_dot_list[:,1],2), label='rho * omega^2 without CBF')
plt.legend()
plt.show()


print(">> End of the script")