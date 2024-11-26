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
m = 50  # number of points between control points
waypoints = np.array([[-2,2],[-0.53,1.71],[0.81,1],[-0.76,0.30],[-1.45,-0.70],[-0.43,-1.21],[1.26,-1.29],[3.00,-1.0]])
ref_path, ref_vel = bz.evaluate_bezier(waypoints, m)  # evaluate Interpolating Bezier curves

plt.plot(ref_path[:,0], ref_path[:,1],'b-',label='Reference trajectory')
plt.plot(ref_path[0,0], ref_path[0,1],'ko',label='Start')
plt.plot(ref_path[-1,0], ref_path[-1,1],'kx',label='Goal')
plt.title('Ref. training trajectory')
plt.legend()
plt.show()

# DMPs training
n_bfs = 100  # number of basis functions
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 1000, dt = 0.001,
                              alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs
dmp_traj.imitate_path(x_des=ref_path)  # train the DMPs

# DMPs execution (no CBF)
dmp_traj.x_0 = np.array([-1.5, 1.5])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([3, -1.0])  # new goal in cartesian coordinates
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.x)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.x)  # a_x, a_y
# Loop
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < 0.01:
    x, x_dot, x_ddot = dmp_traj.step()  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# Centrifugal force
F_cf = ((x_list[:,0]*x_dot_list[:,1]-x_list[:,1]*x_dot_list[:,0])**2/((x_list[:,0]**2+x_list[:,1]**2)**(3/2)))


# Plot the result
plt.subplot(2,1,1)
plt.plot(ref_path[:,0], ref_path[:,1],'b--',label='Reference traj.')
plt.plot(ref_path[0,0], ref_path[0,1],'ko',label='x0')
plt.plot(ref_path[-1,0], ref_path[-1,1],'kx',label='xg')
plt.plot(x_list[:,0], x_list[:,1],'r-',label='Learnt traj.')
plt.plot(x_list[0,0], x_list[0,1],'go',label='new x0')
plt.plot(x_list[-1,0], x_list[-1,1],'rx',label='new xg')
plt.plot()
plt.plot()
plt.title('Learnt vs Reference')
plt.legend()

plt.subplot(2,1,2)
plt.plot(F_cf, 'b-',label='Centrifugal force')  # centrifugal force
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g') # static friction (centripeal force)
plt.title('Centrifugal force')
plt.legend()
plt.show()



print(">> End of the script")