import numpy as np
import matplotlib.pyplot as plt
import copy 
import matplotlib.pyplot as plt
from matplotlib import rc
from dmp import dmp
import bezier_interp as bz
from dmp import dmp, obstacle_superquadric as obs
from cbf import CBF
from ddmr import DDMR
import scipy.sparse as sparse

# =============================================================================
# CASE 1: DMP with h(x) = v_max - sqrt(dx^2 + dy^2) as CBF
# =============================================================================

# Reference trajectory (Cartesian coordinates)
N = 1000  # discretization points
a0 = 3.0  # ellipse major axis
a1 = 1.0  # ellipse minor axis
t = np.linspace(0,np.pi,N)  # time
x = a0*np.cos(t)  # x
y = a1*np.sin(t)  # y
dx = -a0*np.sin(t)  # dx
dy = a1*np.cos(t)  # dy
ref_path = np.vstack((x,y)).T  # reference path
ref_vel = np.vstack((dx,dy)).T  # reference velocity

# DMPs training
n_bfs = 100  # number of basis functions
time_step = 0.01  # time-step
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 100, dt = time_step, T = t[-1],
                              alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs (K=115)
dmp_traj.imitate_path(x_des = ref_path)  # train the DMPs

# DMPs execution (no CBF)
dmp_traj.x_0 = np.array([3.0, 0.0])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([-3.0, 0.0])  # new goal in cartesian coordinates
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

# Save the learnt trajectory for the next part
learnt_path = copy.deepcopy(x_list)
learnt_vel = copy.deepcopy(x_dot_list)

# DMPs execution (NO CBF) with different tau
tau = 1.10  # time scaling factor
dmp_traj.x_0 = np.array([3.0, 0.0])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([-2.5, 0.0])  # new goal in cartesian coordinates
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# Loop
goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    x, x_dot, x_ddot = dmp_traj.step(tau = tau)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# Save the learnt trajectory for the next part
path_tau = copy.deepcopy(x_list)
path_vel = copy.deepcopy(x_dot_list) / tau
v_tau = np.sqrt(path_vel[:,0]**2+path_vel[:,1]**2)  # velocity
tVec_tau = np.linspace(0,len(v_tau)*time_step,len(v_tau))  # time vector

# OBSTACLE (NO CBF)
dmp_traj.x_0 = np.array([3.0, 0.0])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([-2.5, 0.0])  # new goal in cartesian coordinates
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

# obstacle_center = np.array(learnt_path[int(3*len(learnt_path)/5)]) + np.array([-0.05,-0.05])  # obstacle center
# obstacle_center = np.array(learnt_path[int(4*len(learnt_path)/7)]) + np.array([-0.05,-0.05])  # obstacle center
# radius = 0.2
# obstacle_axis = np.ones(dmp_traj.n_dmps) * radius

# Superquadric parameters for obstacle
lmbda = 2.0  # gain of relative orientation function (-lambda*cos(theta))
beta = 2.0  # exponent of the relative orientation function (-cos(theta)^beta)
eta = 1.0  # exponent of the superquadric function (C^eta(x))
# obstacle = obs.Obstacle_Dynamic(center = obstacle_center, axis = obstacle_axis, 
#                                 lmbda = lmbda, beta=beta, eta=eta, coeffs = np.ones(dmp_traj.n_dmps))

goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = np.array([0,0])  # no obstacle external force
    x, x_dot, x_ddot = dmp_traj.step(external_force = obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# Save the learnt trajectory for the next part
obs_path_nocbf = copy.deepcopy(x_list)
obs_vel_nocbf = copy.deepcopy(x_dot_list)
v_nocbf = np.sqrt(obs_vel_nocbf[:,0]**2+obs_vel_nocbf[:,1]**2)  # velocity
tVec_nocbf = np.linspace(0,len(v_nocbf)*time_step,len(v_nocbf))  # time vector

# OBSTACLE (CBF)
alpha = 50  # CBF gain
v_max = 2.50  # maximum velocity [m/s]
K_approx = 0.0001  # approximation gain
cbf = CBF()  # CBF initialization
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# obstacle = obs.Obstacle_Dynamic(center = obstacle_center, axis = obstacle_axis, 
#                                 lmbda = lmbda, beta = beta, eta = eta, coeffs = np.ones(dmp_traj.n_dmps))

goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = np.array([0,0])  # no obstacle external force
    external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, v_max = v_max, alpha = alpha, exp = 1.0, 
                                                      obs_force = obs_force, K_appr = K_approx, type = 'velocity')  # compute the external force
    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force + obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# Save the learnt trajectory for the next part
obs_path_cbf = copy.deepcopy(x_list)
obs_vel_cbf = copy.deepcopy(x_dot_list) 
v_cbf = np.sqrt(obs_vel_cbf[:,0]**2+obs_vel_cbf[:,1]**2)  # velocity
tVec_cbf = np.linspace(0,len(v_cbf)*time_step,len(v_cbf))  # time vector

plt.figure(1, figsize=(8, 6), tight_layout=True)
plt.subplots_adjust(hspace=0.3)  # Adjust the space between the subplots
plt.subplot(2,1,1)
plt.plot(obs_path_nocbf[:,0],obs_path_nocbf[:,1],'r-',label = 'no cbf')
plt.plot(obs_path_cbf[:,0],obs_path_cbf[:,1],'b-',label = 'cbf')
plt.plot(path_tau[:,0],path_tau[:,1],'g-',label = r'$\tau = 1.10$')
# circle = plt.Circle(obstacle_center, radius, color='darkgreen', fill=False, linestyle='-', label='obstacle', linewidth = 2)
# plt.gca().add_patch(circle)
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend(loc = 'upper right')

plt.subplot(2,1,2)
plt.plot(tVec_nocbf,v_nocbf,'r-',label = 'no cbf')
plt.plot(tVec_cbf,v_cbf,'b-',label = 'cbf')
plt.plot(tVec_nocbf,v_max*np.ones(len(v_nocbf)),'k--',label = r'$v_{max}$')
plt.plot(tVec_tau,v_tau,'g-',label = r'$\tau = 1.10$')
plt.xlabel('Time [s]')
plt.ylabel(r'$h(x)$')
plt.legend(loc = 'upper right')
#plt.show()

# =============================================================================
# CASE 2: DMPs and h(x) = mu_s*g - (dy*x-dx*y)^2/(x^2+y^2) as CBF
# =============================================================================

# Reference trajectory (Cartesian coordinates) for inward spiral
N = 1000  # discretization points
a0 = 3.0  # ellipse major axis
a1 = 1.0  # ellipse minor axis
t = np.linspace(0,np.pi,N)  # time
x = a0*np.cos(t)  # x
y = a1*np.sin(t)  # y
dx = -a0*np.sin(t)  # dx
dy = a1*np.cos(t)  # dy
ref_path = np.vstack((x,y)).T  # reference path
ref_vel = np.vstack((dx,dy)).T  # reference velocity

# DMPs training
n_bfs = 100  # number of basis functions
time_step = 0.01  # time-step
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 100, dt = time_step, T = t[-1],
                              alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs (K=115)
dmp_traj.imitate_path(x_des = ref_path)  # train the DMPs

# DMPs (NO CBF)
dmp_traj.x_0 = np.array([-2.0, 1.5])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([3.0, -1.0])  # new goal in cartesian coordinates
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

# Save the learnt trajectory for the next part
path_nocbf = copy.deepcopy(x_list)
vel_nocbf = copy.deepcopy(x_dot_list)

# Time vector and centrifugal force
tVec_nocbf = np.linspace(0,len(path_nocbf)*time_step,len(path_nocbf))  # time vector
F_nocbf = (path_nocbf[:,0]*vel_nocbf[:,1]-path_nocbf[:,1]*vel_nocbf[:,0])**2/((path_nocbf[:,0]**2+path_nocbf[:,1]**2)**(3/2))

# DMPs (CBF)
alpha = 50  # CBF gain
a_max = 12.0  # maximum acceleration [m/s^2]
K_approx = 0.0001  # approximation gain
cbf = CBF()  # CBF initialization
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = np.array([0,0])  # no obstacle
    external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, a_max = a_max, alpha = alpha, exp = 1.0, 
                                                      obs_force = obs_force, K_appr = K_approx, type = 'force')  # compute the external force
    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force + obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))


# Save the learnt trajectory for the next part
path_cbf = copy.deepcopy(x_list)
vel_cbf = copy.deepcopy(x_dot_list)

# Time vector and centrifugal force
tVec_cbf = np.linspace(0,len(path_nocbf)*time_step,len(path_nocbf))  # time vector
F_cbf = (path_cbf[:,0]*vel_cbf[:,1]-path_cbf[:,1]*vel_cbf[:,0])**2/((path_cbf[:,0]**2+path_cbf[:,1]**2)**(3/2))

plt.figure(2, figsize=(8, 6), tight_layout=True)
plt.subplots_adjust(hspace=0.3)  # Adjust the space between the subplots
plt.subplot(2,1,1)
plt.plot(path_nocbf[:,0],path_nocbf[:,1],'r-',label = 'no cbf')
plt.plot(path_cbf[:,0],path_cbf[:,1],'b--',label = 'cbf')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend(loc = 'lower right')

plt.subplot(2,1,2)  
plt.plot(tVec_nocbf,F_nocbf,'r-',label = 'no cbf')
plt.plot(tVec_cbf,F_cbf,'b-',label = 'cbf')
plt.plot(tVec_nocbf,a_max*np.ones(len(v_nocbf)),'k--',label = r'$a_{max}$')
plt.xlabel('Time [s]')
plt.ylabel(r'$h(x)$')
plt.legend(loc = 'lower right')

# =============================================================================
# CASE 3: DMPs with obstacles as CBF
# =============================================================================
# Reference trajectory (Cartesian coordinates)
N = 1000  # discretization points
a0 = 3.0  # ellipse major axis
a1 = 1.0  # ellipse minor axis
t = np.linspace(0,np.pi,N)  # time
x = a0*np.cos(t)  # x
y = a1*np.sin(t)  # y
dx = -a0*np.sin(t)  # dx
dy = a1*np.cos(t)  # dy
ref_path = np.vstack((x,y)).T  # reference path
ref_vel = np.vstack((dx,dy)).T  # reference velocity

# DMPs training
n_bfs = 100  # number of basis functions
time_step = 0.01  # time-step
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 100, dt = time_step, T = t[-1],
                              alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs (K=115)
dmp_traj.imitate_path(x_des = ref_path)  # train the DMPs

# DMPs execution (no CBF)
dmp_traj.x_0 = np.array([3.0, 0.0])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([-2.5, 0.0])  # new goal in cartesian coordinates
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

# Save the learnt trajectory for the next part
learnt_path = copy.deepcopy(x_list)
learnt_vel = copy.deepcopy(x_dot_list)

# DMPs with Obstacle as CBF
alpha = 50  # CBF gain
cbf = CBF()  # CBF initialization
delta_0 = 0.05  # small constant for control barrier function
eta = 0.25  # repulsive gain factor
r_min = 0.25  # radius over which the repulsive potential field is active
gamma = 100.0  # maximum acceleration for the robot
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
obstacle_center = np.array([0.00, 0.90])  # obstacle center

goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = np.array([0,0])  # no obstacle external force
    external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha = alpha, exp = 1.0, delta_0 = delta_0, eta = eta, r_min = r_min, gamma = gamma, obs_center = obstacle_center,
                                                      obs_force = obs_force, K_appr = K_approx, type = 'obstacle')  # compute the external force
    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force + obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# Save the learnt trajectory for the next part
path_cbf = copy.deepcopy(x_list)
vel_cbf = copy.deepcopy(x_dot_list)

plt.figure(3, figsize=(8, 6), tight_layout=True)
plt.plot(learnt_path[:,0],learnt_path[:,1],'b--',label = 'learnt path')
plt.plot(path_cbf[:,0],path_cbf[:,1],'b-',label = 'cbf')
plt.plot(obstacle_center[0],obstacle_center[1],'ro',label = 'obstacle')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend(loc = 'lower right')

plt.show()
print(">>  End of the script")