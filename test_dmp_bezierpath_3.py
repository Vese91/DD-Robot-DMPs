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
# Parametric polynomial of degree 3
N = 1000  # number of discretization points
t = np.linspace(0,2,N)  # time vector
x = t
y = (1-t)**2
dx = np.ones(N)
dy = -2*(1-t)
ref_path = np.vstack((x,y)).T  # reference path
ref_vel = np.vstack((dx,dy)).T  # reference velocity

plt.plot(ref_path[:,0], ref_path[:,1],'b-',label='Reference trajectory')
plt.plot(ref_path[0,0], ref_path[0,1],'ko',label='Start')
plt.plot(ref_path[-1,0], ref_path[-1,1],'kx',label='Goal')
plt.title('Ref. training trajectory')
plt.legend()
plt.show()

# DMPs training
n_bfs = 100  # number of basis functions
Tf = t[-1]  # duration
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 200, dt = 0.01, T = Tf,
                              alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs
dmp_traj.imitate_path(x_des=ref_path)  # train the DMPs

# DMPs execution (no CBF)
dmp_traj.x_0 = np.array([0.0, 1.0])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([1.00, 1.00])  # new goal in cartesian coordinates
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

# Centrifugal force
F_cf1 = (ref_path[:,0]*ref_vel[:,1]-ref_path[:,1]*ref_vel[:,0])**2/((ref_path[:,0]**2+ref_path[:,1]**2)**(3/2))  # centrifugal force in ref path
F_cf2 = ((x_list[:,0]*x_dot_list[:,1]-x_list[:,1]*x_dot_list[:,0])**2/((x_list[:,0]**2+x_list[:,1]**2)**(3/2)))  # centrifugal force in learnt path

# Plot the result
plt.plot(ref_path[:,0], ref_path[:,1],'r--',label='Reference traj.')
plt.plot(ref_path[0,0], ref_path[0,1],'ko',label='x0')
plt.plot(ref_path[-1,0], ref_path[-1,1],'kx',label='xg')
plt.plot(x_list[:,0], x_list[:,1],'b-',label='Learnt traj.')
plt.plot(x_list[0,0], x_list[0,1],'go',label='new x0')
plt.plot(x_list[-1,0], x_list[-1,1],'rx',label='new xg')
#plt.title('Learnt vs Reference (no CBF)')
plt.legend()
plt.show()

plt.subplot(2,1,1)
plt.plot(F_cf1, 'b--',label='Centrifugal force (ref)')  # centrifugal force
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g') # static friction (centripeal force)
plt.legend()
#plt.title('Centrifugal force (no CBF)')

plt.subplot(2,1,2)
plt.plot(F_cf2, 'b-',label='Centrifugal force (learnt)')  # centrifugal force
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g') # static friction (centripeal force)
plt.legend()
plt.show()

# Save the learnt trajectory for the next part
learnt_path = copy.deepcopy(x_list)
learnt_vel = copy.deepcopy(x_dot_list)

# DMPs execution (with CBF)
cbf = CBF()
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# Loop
goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha, mu_s, g, exp)
    x, x_dot, x_ddot = dmp_traj.step(external_force=external_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# Centrifugal force
F_cf1 = (learnt_path[:,0]*learnt_vel[:,1]-learnt_path[:,1]*learnt_vel[:,0])**2/((learnt_path[:,0]**2+learnt_path[:,1]**2)**(3/2))  # centrifugal force in ref path (with CBF)
F_cf2 = ((x_list[:,0]*x_dot_list[:,1]-x_list[:,1]*x_dot_list[:,0])**2/((x_list[:,0]**2+x_list[:,1]**2)**(3/2)))  # centrifugal force in learnt path (with CBF)

# Plot the result
plt.plot(learnt_path[:,0], learnt_path[:,1],'r--',label='Learnt traj.')
plt.plot(learnt_path[0,0], learnt_path[0,1],'ko',label='x0')
plt.plot(learnt_path[-1,0], learnt_path[-1,1],'kx',label='xg')
plt.plot(x_list[:,0], x_list[:,1],'b-',label='Learnt traj. with CBF')
plt.plot(x_list[0,0], x_list[0,1],'go',label='new x0')
plt.plot(x_list[-1,0], x_list[-1,1],'rx',label='new xg')
#plt.title('Learnt vs Learnt (with CBF)')
plt.legend()
plt.show()

plt.subplot(2,1,1)
plt.plot(F_cf1, 'b--',label='Centrifugal force (no CBF)')  # centrifugal force
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g') # static friction (centripeal force)
plt.legend()
#plt.title('Centrifugal force (with CBF)')

plt.subplot(2,1,2)
plt.plot(F_cf2, 'b-',label='Centrifugal force (with CBF)')  # centrifugal force
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g') # static friction (centripeal force)
plt.legend()
plt.show()

plt.subplot(2,1,1)
plt.plot(x_dot_list[:,0],'b-',label = 'dx (with CBF)')
plt.plot(learnt_vel[:,0],'r-',label = 'dx (no CBF)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(x_dot_list[:,1],'b-',label = 'dy (with CBF)')
plt.plot(learnt_vel[:,1],'r-',label = 'dy (no CBF)')
plt.legend()
plt.show()

print(">> First part")


# ROBOT CONTROLS


# OBSTACLE
#
# DMPs execution (with obstacles, no CBF)
dmp_traj.x_0 = np.array([-2, 1.5])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([3, -1.0])  # new goal in cartesian coordinates
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

obstacle_center = np.array([0.0, 0.30])  # obstacle center
radius = 0.15
obstacle_axis = np.ones(dmp_traj.n_dmps) * radius

# superquadric parameters
lmbda = 5.0
beta = 5.0
eta = 5.0

obstacle = obs.Obstacle_Dynamic(center = obstacle_center, axis = obstacle_axis, 
                                lmbda = lmbda, beta=beta, eta=eta, coeffs = np.ones(dmp_traj.n_dmps))
# Loop
goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = obstacle.gen_external_force(dmp_traj.x, dmp_traj.dx)
    x, x_dot, x_ddot = dmp_traj.step(external_force=obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# Centrifugal force
F_cf1 = (ref_path[:,0]*ref_vel[:,1]-ref_path[:,1]*ref_vel[:,0])**2/((ref_path[:,0]**2+ref_path[:,1]**2)**(3/2))  # centrifugal force in ref path
F_cf2 = ((x_list[:,0]*x_dot_list[:,1]-x_list[:,1]*x_dot_list[:,0])**2/((x_list[:,0]**2+x_list[:,1]**2)**(3/2)))  # centrifugal force in learnt path

# Plot the result
plt.plot(learnt_path[:,0], learnt_path[:,1],'r--',label='Learnt traj.')
plt.plot(learnt_path[0,0], learnt_path[0,1],'ko',label='x0')
plt.plot(learnt_path[-1,0], learnt_path[-1,1],'kx',label='xg')
plt.plot(x_list[:,0], x_list[:,1],'b-',label='Learnt traj. with CBF')
plt.plot(x_list[0,0], x_list[0,1],'go',label='new x0')
plt.plot(x_list[-1,0], x_list[-1,1],'rx',label='new xg')
# plot a circle for the obstacle
circle = plt.Circle(obstacle_center, obstacle_axis[0], color='g', fill=False)
plt.gca().add_artist(circle)
plt.title('Learnt vs Learnt (with obstacles, no CBF)')
plt.legend()
plt.show()

plt.subplot(2,1,1)
plt.plot(F_cf1,'b--',label='Centrifugal force (with obstacles, no CBF)')  # centrifugal force
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g') # static friction (centripeal force)
plt.legend()
plt.title('Centrifugal force (with obstacles, no CBF)')

plt.subplot(2,1,2)
plt.plot(F_cf2,'b-',label='Centrifugal force')  # centrifugal force
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g') # static friction (centripeal force)
plt.legend()
plt.show()


# DMPs execution (with obstacles, with CBF)
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

obstacle = obs.Obstacle_Dynamic(center = obstacle_center, axis = obstacle_axis, lmbda = lmbda, beta=beta, eta=eta, coeffs = np.ones(dmp_traj.n_dmps))
# Loop
goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = obstacle.gen_external_force(dmp_traj.x, dmp_traj.dx)
    external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha, mu_s, g, exp, obs_force)
    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force + obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# Centrifugal force
F_cf1 = (learnt_path[:,0]*learnt_vel[:,1]-learnt_path[:,1]*learnt_vel[:,0])**2/((learnt_path[:,0]**2+learnt_path[:,1]**2)**(3/2))  # centrifugal force in ref path (with CBF)
F_cf2 = ((x_list[:,0]*x_dot_list[:,1]-x_list[:,1]*x_dot_list[:,0])**2/((x_list[:,0]**2+x_list[:,1]**2)**(3/2)))  # centrifugal force in learnt path (with CBF)

# Plot the result
plt.plot(learnt_path[:,0], learnt_path[:,1],'r--',label='Learnt traj.')
plt.plot(learnt_path[0,0], learnt_path[0,1],'ko',label='x0')
plt.plot(learnt_path[-1,0], learnt_path[-1,1],'kx',label='xg')
plt.plot(x_list[:,0], x_list[:,1],'b-',label='Learnt traj. with CBF')
plt.plot(x_list[0,0], x_list[0,1],'go',label='new x0')
plt.plot(x_list[-1,0], x_list[-1,1],'rx',label='new xg')
# plot a circle for the obstacle
circle = plt.Circle(obstacle_center, obstacle_axis[0], color='g', fill=False)
plt.gca().add_artist(circle)
plt.title('Learnt vs Learnt (with obstacles, with CBF)')
plt.legend()
plt.show()

plt.subplot(2,1,1)
plt.plot(F_cf1, 'b--',label='Centrifugal force (with obstacles, with CBF)')  # centrifugal force
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g') # static friction (centripeal force)
plt.legend()
plt.title('Centrifugal force (with obstacles, with CBF)')

plt.subplot(2,1,2)
plt.plot(F_cf2, 'b-',label='Centrifugal force')  # centrifugal force
plt.axhline(y = mu_s * g, color='r', linestyle='-', label='mu_s * g') # static friction (centripeal force)
plt.legend()
plt.show()

plt.subplot(2,1,1)
plt.plot(x_dot_list[:,0],'b-',label = 'dx (with CBF)')
plt.plot(learnt_vel[:,0],'r-',label = 'dx (no CBF)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(x_dot_list[:,1],'b-',label = 'dy (with CBF)')
plt.plot(learnt_vel[:,1],'r-',label = 'dy (no CBF)')
plt.legend()
plt.show()




print(">> End of the script")