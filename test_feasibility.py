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

def gen_dynamic_force(gamma, eta, v_ego, v_obs, p_obs, p_ego):
    """
    From the paper: "Duhé, J. F., Victor, S., & Melchior, P. (2021). Contributions on artificial potential field method for effective obstacle avoidance. Fractional Calculus and Applied Analysis, 24, 421-446.

    a_max: maximum acceleration
    eta: repulsive gain factor
    v_ego: velocity of the ego vehicle
    v_obs: velocity of the obstacle
    p_obs: position of the obstacle
    p_ego: position of the ego vehicle
    """
    v_rel = v_ego - v_obs
    p_rel = p_ego - p_obs
    rho_s = np.linalg.norm(p_rel)
    v_ro = np.dot(v_rel, -p_rel) / rho_s
    rho_m = v_ro**2 / (2 * gamma)
    rho_delta = rho_s - rho_m
    v_ro_orth = np.sqrt(np.linalg.norm(v_rel)**2 - v_ro**2)
    nabla_p = - eta * (1 + v_ro/gamma) / (rho_delta**2)
    nabla_v = eta * v_ro * v_ro_orth / (rho_delta**2) / (gamma * rho_s)
    return np.array([- nabla_p, - nabla_v])


def gen_potential(gamma, eta, v_ego, v_obs, p_obs, p_ego):
    """
    From the paper: "Duhé, J. F., Victor, S., & Melchior, P. (2021). Contributions on artificial potential field method for effective obstacle avoidance. Fractional Calculus and Applied Analysis, 24, 421-446.

    a_max: maximum acceleration
    eta: repulsive gain factor
    v_ego: velocity of the ego vehicle
    v_obs: velocity of the obstacle
    p_obs: position of the obstacle
    p_ego: position of the ego vehicle
    """
    v_rel = v_ego - v_obs
    p_rel = p_ego - p_obs
    rho_s = np.linalg.norm(p_rel)
    v_ro = np.dot(v_rel, -p_rel) / rho_s
    rho_m = v_ro**2 / (2 * gamma)
    rho_delta = rho_s - rho_m
    U = eta * (1/rho_delta - 1/r_min)
    return -1/(1+U)


# # =============================================================================
# # CASE 1: DMP with h(x) = v_max - sqrt(dx^2 + dy^2) as CBF
# # =============================================================================

# # Reference trajectory (Cartesian coordinates)
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

# # DMPs training
# n_bfs = 100  # number of basis functions
# time_step = 0.01  # time-step
# dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 100, dt = time_step, T = t[-1],
#                               alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs (K=115)
# dmp_traj.imitate_path(x_des = ref_path)  # train the DMPs

# # DMPs execution (no CBF)
# dmp_traj.x_0 = np.array([3.0, 0.0])  # new start in cartesian coordinates
# dmp_traj.x_goal = np.array([-3.0, 0.0])  # new goal in cartesian coordinates
# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list = np.array(dmp_traj.x) # x, y
# x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# # Loop
# goal_tol = 0.01 # goal tolerance
# while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
#     x, x_dot, x_ddot = dmp_traj.step()  # execute the DMPs
#     x_list = np.vstack((x_list, x))
#     x_dot_list = np.vstack((x_dot_list, x_dot))
#     x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# # Save the learnt trajectory for the next part
# learnt_path = copy.deepcopy(x_list)
# learnt_vel = copy.deepcopy(x_dot_list)

# # OBSTACLE (NO CBF)
# dmp_traj.x_0 = np.array([3.0, 0.0])  # new start in cartesian coordinates
# dmp_traj.x_goal = np.array([-2.5, 0.0])  # new goal in cartesian coordinates
# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list = np.array(dmp_traj.x) # x, y
# x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

# # obstacle_center = np.array(learnt_path[int(3*len(learnt_path)/5)]) + np.array([-0.05,-0.05])  # obstacle center
# # obstacle_center = np.array(learnt_path[int(4*len(learnt_path)/7)]) + np.array([-0.05,-0.05])  # obstacle center
# # radius = 0.2
# # obstacle_axis = np.ones(dmp_traj.n_dmps) * radius

# # Superquadric parameters for obstacle
# lmbda = 2.0  # gain of relative orientation function (-lambda*cos(theta))
# beta = 2.0  # exponent of the relative orientation function (-cos(theta)^beta)
# eta = 1.0  # exponent of the superquadric function (C^eta(x))
# # obstacle = obs.Obstacle_Dynamic(center = obstacle_center, axis = obstacle_axis, 
# #                                 lmbda = lmbda, beta=beta, eta=eta, coeffs = np.ones(dmp_traj.n_dmps))

# goal_tol = 0.01 # goal tolerance
# while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
#     obs_force = np.array([0,0])  # no obstacle external force
#     x, x_dot, x_ddot = dmp_traj.step(external_force=obs_force)  # execute the DMPs
#     x_list = np.vstack((x_list, x))
#     x_dot_list = np.vstack((x_dot_list, x_dot))
#     x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# # Save the learnt trajectory for the next part
# obs_path_nocbf = copy.deepcopy(x_list)
# obs_vel_nocbf = copy.deepcopy(x_dot_list)
# v_nocbf = np.sqrt(obs_vel_nocbf[:,0]**2+obs_vel_nocbf[:,1]**2)  # velocity
# tVec_nocbf = np.linspace(0,len(v_nocbf)*time_step,len(v_nocbf))  # time vector

# # OBSTACLE (CBF)
# alpha = 50  # CBF gain
# v_max = 2.50  # maximum velocity [m/s]
# K_approx = 0.0001  # approximation gain
# cbf = CBF()  # CBF initialization
# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list = np.array(dmp_traj.x) # x, y
# x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# # obstacle = obs.Obstacle_Dynamic(center = obstacle_center, axis = obstacle_axis, 
# #                                 lmbda = lmbda, beta = beta, eta = eta, coeffs = np.ones(dmp_traj.n_dmps))

# goal_tol = 0.01 # goal tolerance
# while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
#     obs_force = np.array([0,0])  # no obstacle external force
#     external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, v_max = v_max, alpha = alpha, exp = 1.0, 
#                                                       obs_force = obs_force, K_appr = K_approx, type = 'velocity')  # compute the external force
#     x, x_dot, x_ddot = dmp_traj.step(external_force = external_force + obs_force)  # execute the DMPs
#     x_list = np.vstack((x_list, x))
#     x_dot_list = np.vstack((x_dot_list, x_dot))
#     x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# # Save the learnt trajectory for the next part
# obs_path_cbf = copy.deepcopy(x_list)
# obs_vel_cbf = copy.deepcopy(x_dot_list)
# v_cbf = np.sqrt(obs_vel_cbf[:,0]**2+obs_vel_cbf[:,1]**2)  # velocity
# tVec_cbf = np.linspace(0,len(v_cbf)*time_step,len(v_cbf))  # time vector

# plt.figure(1, figsize=(8, 6), tight_layout=True)
# plt.subplots_adjust(hspace=0.3)  # Adjust the space between the subplots
# plt.subplot(2,1,1)
# plt.plot(obs_path_nocbf[:,0],obs_path_nocbf[:,1],'r-',label = 'no cbf')
# plt.plot(obs_path_cbf[:,0],obs_path_cbf[:,1],'b-',label = 'cbf')
# # circle = plt.Circle(obstacle_center, radius, color='darkgreen', fill=False, linestyle='-', label='obstacle', linewidth = 2)
# # plt.gca().add_patch(circle)
# plt.xlabel('$x$ [m]')
# plt.ylabel('$y$ [m]')
# plt.legend(loc = 'lower right')

# plt.subplot(2,1,2)
# plt.plot(tVec_nocbf,v_nocbf,'r-',label = 'no cbf')
# plt.plot(tVec_cbf,v_cbf,'b-',label = 'cbf')
# plt.plot(tVec_nocbf,v_max*np.ones(len(v_nocbf)),'k--',label = r'$v_{max}$')
# plt.xlabel('Time [s]')
# plt.ylabel(r'$h(x)$')
# plt.legend(loc = 'lower right')
# #plt.show()

# # =============================================================================
# # CASE 2: DMPs and h(x) = mu_s*g - (dy*x-dx*y)^2/(x^2+y^2) as CBF
# # =============================================================================

# # Reference trajectory (Cartesian coordinates) for inward spiral
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

# # DMPs training
# n_bfs = 100  # number of basis functions
# time_step = 0.01  # time-step
# dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 100, dt = time_step, T = t[-1],
#                               alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs (K=115)
# dmp_traj.imitate_path(x_des = ref_path)  # train the DMPs

# # DMPs (NO CBF)
# dmp_traj.x_0 = np.array([-2.0, 1.5])  # new start in cartesian coordinates
# dmp_traj.x_goal = np.array([3.0, -1.0])  # new goal in cartesian coordinates
# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list = np.array(dmp_traj.x) # x, y
# x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# # Loop
# goal_tol = 0.01 # goal tolerance
# while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
#     x, x_dot, x_ddot = dmp_traj.step()  # execute the DMPs
#     x_list = np.vstack((x_list, x))
#     x_dot_list = np.vstack((x_dot_list, x_dot))
#     x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# # Save the learnt trajectory for the next part
# path_nocbf = copy.deepcopy(x_list)
# vel_nocbf = copy.deepcopy(x_dot_list)

# # Time vector and centrifugal force
# tVec_nocbf = np.linspace(0,len(path_nocbf)*time_step,len(path_nocbf))  # time vector
# F_nocbf = (path_nocbf[:,0]*vel_nocbf[:,1]-path_nocbf[:,1]*vel_nocbf[:,0])**2/((path_nocbf[:,0]**2+path_nocbf[:,1]**2)**(3/2))

# # DMPs (CBF)
# alpha = 50  # CBF gain
# a_max = 12.0  # maximum acceleration [m/s^2]
# K_approx = 0.0001  # approximation gain
# cbf = CBF()  # CBF initialization
# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list = np.array(dmp_traj.x) # x, y
# x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

# goal_tol = 0.01 # goal tolerance
# while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
#     obs_force = np.array([0,0])  # no obstacle
#     external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, a_max = a_max, alpha = alpha, exp = 1.0, 
#                                                       obs_force = obs_force, K_appr = K_approx, type = 'force')  # compute the external force
#     x, x_dot, x_ddot = dmp_traj.step(external_force = external_force + obs_force)  # execute the DMPs
#     x_list = np.vstack((x_list, x))
#     x_dot_list = np.vstack((x_dot_list, x_dot))
#     x_ddot_list = np.vstack((x_ddot_list, x_ddot))


# # Save the learnt trajectory for the next part
# path_cbf = copy.deepcopy(x_list)
# vel_cbf = copy.deepcopy(x_dot_list)

# # Time vector and centrifugal force
# tVec_cbf = np.linspace(0,len(path_nocbf)*time_step,len(path_nocbf))  # time vector
# F_cbf = (path_cbf[:,0]*vel_cbf[:,1]-path_cbf[:,1]*vel_cbf[:,0])**2/((path_cbf[:,0]**2+path_cbf[:,1]**2)**(3/2))

# plt.figure(2, figsize=(8, 6), tight_layout=True)
# plt.subplots_adjust(hspace=0.3)  # Adjust the space between the subplots
# plt.subplot(2,1,1)
# plt.plot(path_nocbf[:,0],path_nocbf[:,1],'r-',label = 'no cbf')
# plt.plot(path_cbf[:,0],path_cbf[:,1],'b--',label = 'cbf')
# plt.xlabel('$x$ [m]')
# plt.ylabel('$y$ [m]')
# plt.legend(loc = 'lower right')

# plt.subplot(2,1,2)  
# plt.plot(tVec_nocbf,F_nocbf,'r-',label = 'no cbf')
# plt.plot(tVec_cbf,F_cbf,'b-',label = 'cbf')
# plt.plot(tVec_nocbf,a_max*np.ones(len(v_nocbf)),'k--',label = r'$a_{max}$')
# plt.xlabel('Time [s]')
# plt.ylabel(r'$h(x)$')
# plt.legend(loc = 'lower right')

# # =============================================================================
# # CASE 1: DMP with h(x) = v_max - sqrt(dx^2 + dy^2) as CBF
# # =============================================================================

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

# DMPs execution (no CBF) tau = 1.1
tau = 1.10  # temporal scaling factor
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
vel_tau = copy.deepcopy(x_dot_list)
v_tau = np.sqrt(vel_tau[:,0]**2+vel_tau[:,1]**2)/tau  # velocity
tVec_tau = np.linspace(0,len(v_tau)*time_step,len(v_tau))  # time vector

# OBSTACLE (NO CBF)
dmp_traj.x_0 = np.array([3.0, 0.0])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([-2.5, 0.0])  # new goal in cartesian coordinates
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

# Superquadric parameters for obstacle
lmbda = 2.0  # gain of relative orientation function (-lambda*cos(theta))
beta = 2.0  # exponent of the relative orientation function (-cos(theta)^beta)
eta = 1.0  # exponent of the superquadric function (C^eta(x))
# obstacle = obs.Obstacle_Dynamic(center = obstacle_center, axis = obstacle_axis, 
#                                 lmbda = lmbda, beta=beta, eta=eta, coeffs = np.ones(dmp_traj.n_dmps))

goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = np.array([0,0])  # no obstacle external force
    x, x_dot, x_ddot = dmp_traj.step(external_force=obs_force)  # execute the DMPs
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
plt.plot(obs_path_nocbf[:,0],obs_path_nocbf[:,1],'r-',label = 'DMP')
plt.plot(obs_path_cbf[:,0],obs_path_cbf[:,1],'b-',label = 'CMP')
plt.plot(path_tau[:,0],path_tau[:,1],'g-',label = r'$\tau = 1.1$')
plt.plot(obs_path_nocbf[-1,0],obs_path_nocbf[-1,1],'gx',label = 'Goal')
plt.plot(obs_path_nocbf[0,0],obs_path_nocbf[0,1],'bo',label = 'Start')
plt.xlabel('$x$ [m]', fontsize = 14)
plt.ylabel('$y$ [m]', fontsize = 14)
plt.legend(loc = 'upper right')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(tVec_nocbf,v_nocbf,'r-',label = 'DMP')
plt.plot(tVec_cbf,v_cbf,'b-',label = 'CMP')
plt.plot(tVec_nocbf,v_max*np.ones(len(v_nocbf)),'k--',label = r'$v_{max}$')
plt.plot(tVec_tau,v_tau,'g-',label = r'$\tau = 1.1$')
plt.xlabel('Time [s]', fontsize = 14)
plt.ylabel(r'$v\,(t)$', fontsize = 14)
plt.legend(loc = 'upper right')
plt.grid(True)
#plt.show()

# # Animation for RAL video
# import matplotlib.animation as animation

# tb_vmax = v_max / 12.5  # rescale v_max for turtlebot 3
# tb_vcbf = v_cbf / 12.5  # rescale v_cbf for turtlebot 3
# tVec_vmax = tVec_nocbf * 5.73  # time vector for v_max

# fig, ax = plt.subplots(figsize=(16, 9))  # Set aspect ratio to 16:9
# line, = ax.plot([], [], 'b-', label='CMP')
# ax.plot(tVec_vmax, tb_vmax * np.ones(len(v_nocbf)), 'k--', label=r'$v_{max}$', linewidth=2)
# ax.set_ylabel('$h(x)$ [m/s]', fontsize=18)
# ax.set_xlabel('Time [s]', fontsize=18)
# ax.legend(loc='upper right', fontsize=18)
# plt.xlim([0, 3.16*5.73])
# plt.ylim([0, 0.25])
# ax.grid(True)

# def init():
#     line.set_data([], [])
#     return line,

# def update(frame):
#     line.set_data(tVec_vmax[:frame], tb_vcbf[:frame])
#     line.set_linewidth(3)  # Set the line width to 2
#     return line,

# ani = animation.FuncAnimation(fig, update, frames=len(tVec_cbf), init_func=init, blit=True, interval=20, repeat=False)
# plt.show()

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
plt.plot(path_nocbf[:,0],path_nocbf[:,1],'r-',label = 'DMP')
plt.plot(path_cbf[:,0],path_cbf[:,1],'b-',label = 'CMP')
plt.plot(path_nocbf[-1,0],path_nocbf[-1,1],'gx',label = 'Goal')
plt.plot(path_nocbf[0,0],path_nocbf[0,1],'bo',label = 'Start')
plt.xlabel('$x$ [m]', fontsize = 14)
plt.ylabel('$y$ [m]', fontsize = 14)
plt.legend(loc = 'upper right')
plt.grid(True)

plt.subplot(2,1,2)  
plt.plot(tVec_nocbf,F_nocbf,'r-',label = 'DMP')
plt.plot(tVec_cbf,F_cbf,'b-',label = 'CMP')
plt.plot(tVec_nocbf,a_max*np.ones(len(v_nocbf)),'k--',label = r'$a_{max}$')
plt.xlabel('Time [s]', fontsize = 14)
plt.ylabel(r'$a\,(t)$', fontsize = 14)
plt.legend(loc = 'lower right')
plt.grid(True)
#plt.show()

#print('>>  End of the first part of the code')

# Animation for RAL video
# a_max = a_max / 12.5  # rescale a_max for turtlebot 3
# F_cbf = F_cbf / 12.5  # rescale F_cbf for turtlebot 3
# tVec_amax = tVec_cbf * 5.43  # time vector for v_max

# import matplotlib.animation as animation

# fig, ax = plt.subplots(figsize=(16, 9))  # Set aspect ratio to 16:9
# line, = ax.plot([], [], 'b-', label='CMP')
# ax.plot(tVec_amax, a_max * np.ones(len(v_nocbf)), 'k--', label=r'$a_{max}$')
# ax.set_ylabel('$h(x)$ [m/s^2]',fontsize=18)
# ax.set_xlabel('Time [s]',fontsize=18)
# ax.legend(loc='upper right',fontsize=18)
# plt.xlim([0, 3.16*5.43])
# plt.ylim([0, 1.10])
# ax.grid(True)

# def init():
#     line.set_data([], [])
#     return line,

# def update(frame):
#     line.set_data(tVec_amax[:frame], F_cbf[:frame])
#     return line,

# ani = animation.FuncAnimation(fig, update, frames=len(tVec_amax), init_func=init, blit=True, interval=20, repeat=False)
# plt.show()

# =============================================================================
# CASE 3: DMPs with obstacles as CBF
# =============================================================================
K_approx = 0.0001  # approximation gain

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
eta = 0.05 # repulsive gain factor (default 0.05)
r_min = 0.25  # radius over which the repulsive potential field is active
gamma = 120.0  # maximum acceleration for the robot (default 100.0)
# a_max = 100.0  # maximum acceleration [m/s^2]
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
obstacle_centers = []  # obstacle center list
num_obstacles = 5  # number of obstacles
num_obst_points = 4  # number of points to define the obstacle
radius = 0.01
for i in range(num_obstacles):
    for j in range(num_obst_points):
        obstacle_centers.append(learnt_path[int(int((i+1)*len(learnt_path)/(num_obstacles+1)))] + radius * np.array([np.cos(2*np.pi*j/float(num_obst_points)), np.sin(2*np.pi*j/float(num_obst_points))]))  # obstacle center

# obstacle_center = np.array([0.00, 0.90])  # obstacle center

goal_tol = 0.01 # goal tolerance
obs_force = np.array([0.,0.])  # no obstacle external force
potentials_cbf = []
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    potential = 0.
    external_force_total = np.array([0.,0.])
    for obstacle_center in obstacle_centers:
        external_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha = alpha, exp = 1.0, delta_0 = delta_0, eta = eta, r_min = r_min, gamma = gamma, obs_center = obstacle_center,
                                                      obs_force = obs_force, K_appr = K_approx, type = 'obstacle')
        external_force_total += external_force
    # external_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha = alpha, exp = 1.0, delta_0 = delta_0, eta = eta, r_min = r_min, gamma = gamma, obs_center = obstacle_centers[0],
                                                    #   obs_force = obs_force, K_appr = K_approx, type = 'obstacle')  # compute the external force
        potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force_total + obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))
    potentials_cbf.append(potential)


# Save the learnt trajectory for the next part
path_cbf = copy.deepcopy(x_list)
vel_cbf = copy.deepcopy(x_dot_list)

dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

goal_tol = 0.01 # goal tolerance
potentials = []
step = 0
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = np.array([0.,0.])
    potential = 0.
    for obstacle_center in obstacle_centers:
        obs_force += gen_dynamic_force(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
        potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
        #print(obs_force)
        #print(step)
    
    x, x_dot, x_ddot = dmp_traj.step(external_force = obs_force)  # execute the DMPs
    step = step + 1
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))
    potentials.append(potential)

plt.figure(3, figsize=(8, 6), tight_layout=True)
plt.subplots_adjust(hspace=0.3)  # Adjust the space between the subplots
plt.plot(x_list[:,0],x_list[:,1],'r-',label = 'DMP')
plt.plot(path_cbf[:,0],path_cbf[:,1],'b-',label = 'CMP')
plt.plot(learnt_path[:,0],learnt_path[:,1],'--',label = 'ref.')
plt.plot(dmp_traj.x_goal[0],dmp_traj.x_goal[1],'gx')
plt.plot(dmp_traj.x_0[0],dmp_traj.x_0[1],'bo',label = 'start')
n_o = 0
for obstacle_center in obstacle_centers:
    if n_o == 0:
        plt.plot(obstacle_center[0],obstacle_center[1],'yo',label = 'obstacle')
    else:
        plt.plot(obstacle_center[0],obstacle_center[1],'yo')
    n_o = n_o + 1
plt.xlabel('$x$ [m]', fontsize = 14)
plt.ylabel('$y$ [m]', fontsize = 14)
plt.legend(loc = 'upper right')
plt.grid(True)

# Save the learnt trajectory for the next part
path_cbf = copy.deepcopy(x_list)
vel_cbf = copy.deepcopy(x_dot_list)

# =============================================================================
# CASE 4: DMPs with obstacles as CBF (narrow passage)
# =============================================================================
K_approx = 0.0001  # approximation gain

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
eta = 0.05 # repulsive gain factor (default 0.05)
r_min = 0.25  # radius over which the repulsive potential field is active
gamma = 100.0  # maximum acceleration for the robot (default 100.0)
# a_max = 100.0  # maximum acceleration [m/s^2]
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
obstacle_centers = []  # obstacle center list
num_obstacles = 10  # number of obstacles
num_obst_points = 4  # number of points to define the obstacle
radius = 0.01
# Place obstacles on one side of the corridor
for i in range(int(np.round(num_obstacles/2))):
    index = min(int((i+1)*len(learnt_path)/6), len(learnt_path)-1)  # Ensure index is within bounds
    obstacle_centers.append(learnt_path[index] + np.array([0, 0.05]))  # offset upward

# Place obstacles on the other side of the corridor
for i in range(int(np.round(num_obstacles/2))):
    index = min(int((i+1)*len(learnt_path)/6), len(learnt_path)-1)  # Ensure index is within bounds
    obstacle_centers.append(learnt_path[index] + np.array([0, -0.05]))  # offset downward

goal_tol = 0.01 # goal tolerance
obs_force = np.array([0.,0.])  # no obstacle external force
potentials_cbf = []
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    potential = 0.
    external_force_total = np.array([0.,0.])
    for obstacle_center in obstacle_centers:
        external_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha = alpha, exp = 1.0, delta_0 = delta_0, eta = eta, r_min = r_min, gamma = gamma, obs_center = obstacle_center,
                                                      obs_force = obs_force, K_appr = K_approx, type = 'obstacle')
        external_force_total += external_force
        potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)

    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force_total + obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))
    potentials_cbf.append(potential)

# Save the learnt trajectory for the next part
path_cbf = copy.deepcopy(x_list)
vel_cbf = copy.deepcopy(x_dot_list)

plt.figure(4, figsize=(8, 6), tight_layout=True)
plt.subplots_adjust(hspace=0.3)  # Adjust the space between the subplots

dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

goal_tol = 0.01 # goal tolerance
potentials = []
step = 0
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = np.array([0.,0.])
    potential = 0.
    for obstacle_center in obstacle_centers:
        obs_force += gen_dynamic_force(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
        potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
        #print(obs_force)
        #print(step)

    x, x_dot, x_ddot = dmp_traj.step(external_force = obs_force)  # execute the DMPs
    step = step + 1
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))
    potentials.append(potential)

plt.plot(x_list[:,0],x_list[:,1],'r-',label = 'DMP')
plt.plot(path_cbf[:,0],path_cbf[:,1],'b-',label = 'CMP')
plt.plot(learnt_path[:,0],learnt_path[:,1],'--',label = 'ref.')
plt.plot(dmp_traj.x_goal[0],dmp_traj.x_goal[1],'gx')
plt.plot(dmp_traj.x_0[0],dmp_traj.x_0[1],'bo',label = 'start')
n_o = 0
for obstacle_center in obstacle_centers:
    if n_o == 0:
        plt.plot(obstacle_center[0],obstacle_center[1],'yo',label = 'obstacle')
    else:
        plt.plot(obstacle_center[0],obstacle_center[1],'yo')
    n_o = n_o + 1
plt.xlabel('$x$ [m]', fontsize = 14)
plt.ylabel('$y$ [m]', fontsize = 14)
plt.legend(loc = 'upper right')
plt.grid(True)

# Save the learnt trajectory for the next part
path_cbf = copy.deepcopy(x_list)
vel_cbf = copy.deepcopy(x_dot_list)


# =============================================================================
# CASE 5: DMPs with obstacles as CBF (moving obstacle)
# =============================================================================
K_approx = 0.0001  # approximation gain

# Reference trajectory (Cartesian coordinates)
N = 1000  # discretization points
a0 = 3.0  # ellipse major axis
a1 = 1.0  # ellipse minor axis
t = np.linspace(0, np.pi, N)  # time
x = a0 * np.cos(t)  # x
y = a1 * np.sin(t)  # y
dx = -a0 * np.sin(t)  # dx
dy = a1 * np.cos(t)  # dy
ref_path = np.vstack((x, y)).T  # reference path
ref_vel = np.vstack((dx, dy)).T  # reference velocity

# DMPs training
n_bfs = 100  # number of basis functions
time_step = 0.01  # time-step
dmp_traj = dmp.DMPs_cartesian(n_dmps=2, n_bfs=n_bfs, K=100, dt=time_step, T=t[-1],
                              alpha_s=2.0, tol=3.0 / 100, rescale="rotodilatation", basis="gaussian")  # set up the DMPs (K=115)
dmp_traj.imitate_path(x_des=ref_path)  # train the DMPs

# DMPs execution (no CBF)
dmp_traj.x_0 = np.array([3.0, 0.0])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([-2.5, 0.0])  # new goal in cartesian coordinates
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x)  # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# Loop
goal_tol = 0.01  # goal tolerance
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
eta = 0.05  # repulsive gain factor (default 0.05)
r_min = 0.25  # radius over which the repulsive potential field is active
gamma = 120.0  # maximum acceleration for the robot (default 100.0)
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x)  # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
obstacle_centers = []  # obstacle center list
num_obstacles = 10  # number of obstacles
num_obst_points = 1  # number of points to define the obstacle
radius = 0.01
for i in range(num_obstacles):
    for j in range(num_obst_points):
        obstacle_centers.append(learnt_path[int(int((i+1)*len(learnt_path)/(num_obstacles+1)))] + radius * np.array([np.cos(2*np.pi*j/float(num_obst_points)), np.sin(2*np.pi*j/float(num_obst_points))]))  # obstacle center

goal_tol = 0.01 # goal tolerance
obs_force = np.array([0.,0.])  # no obstacle external force
potentials_cbf = []
A = 0.01 # amplitude of the moving obstacle
omega = np.pi/2 # frequency of the moving obstacle
t = 0.0  # initial time
Ts = 0.01  # time step for the moving obstacle
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    potential = 0.
    external_force_total = np.array([0.,0.])
    for obstacle_center in obstacle_centers:
        external_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha = alpha, exp = 1.0, delta_0 = delta_0, eta = eta, r_min = r_min, gamma = gamma, obs_center = obstacle_center,
                                                      obs_force = obs_force, K_appr = K_approx, type = 'obstacle')
        external_force_total += external_force
        potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)

        # Update the obstacle center position based on the sine wave
        obstacle_center[0] += A*np.sin(omega*t)  # x position of the moving obstacle
        obstacle_center[1] += A*np.cos(omega*t)  # y position of the moving obstacle
        t += Ts  # increment time

    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force_total + obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))
    potentials_cbf.append(potential)

# Save the learnt trajectory for the next part
path_cbf = copy.deepcopy(x_list)
vel_cbf = copy.deepcopy(x_dot_list)

plt.figure(5, figsize=(8, 6), tight_layout=True)
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

# reset the obstacle centers to the original position
obstacle_centers = []  # obstacle center list
num_obstacles = 10  # number of obstacles
num_obst_points = 1  # number of points to define the obstacle
radius = 0.01
for i in range(num_obstacles):
    for j in range(num_obst_points):
        obstacle_centers.append(learnt_path[int(int((i+1)*len(learnt_path)/(num_obstacles+1)))] + radius * np.array([np.cos(2*np.pi*j/float(num_obst_points)), np.sin(2*np.pi*j/float(num_obst_points))]))  # obstacle center

goal_tol = 0.01 # goal tolerance
potentials = []
step = 0
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = np.array([0.,0.])
    potential = 0.
    for obstacle_center in obstacle_centers:
        obs_force += gen_dynamic_force(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
        potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)

        # Update the obstacle center position based on the sine wave
        obstacle_center[0] += A*np.sin(omega*t)  # x position of the moving obstacle
        obstacle_center[1] += A*np.cos(omega*t)  # y position of the moving obstacle
        plt.plot(obstacle_center[0],obstacle_center[1],'yo')
        t += Ts  # increment time
        
    x, x_dot, x_ddot = dmp_traj.step(external_force = obs_force)  # execute the DMPs
    step = step + 1
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))
    potentials.append(potential)

plt.subplots_adjust(hspace=0.3)  # Adjust the space between the subplots
plt.plot(x_list[:,0],x_list[:,1],'r-',label = 'DMP')
plt.plot(path_cbf[:,0],path_cbf[:,1],'b-',label = 'CMP')
plt.plot(learnt_path[:,0],learnt_path[:,1],'--',label = 'ref.')
plt.plot(dmp_traj.x_goal[0],dmp_traj.x_goal[1],'gx')
plt.plot(dmp_traj.x_0[0],dmp_traj.x_0[1],'bo',label = 'start')
n_o = 0
for obstacle_center in obstacle_centers:
    if n_o == 0:
        plt.plot(obstacle_center[0], obstacle_center[1], 'o', color='darkorange', label='obstacle')
    else:
        plt.plot(obstacle_center[0], obstacle_center[1], 'o', color='darkorange')
    n_o = n_o + 1
plt.xlabel('$x$ [m]', fontsize = 14)
plt.ylabel('$y$ [m]', fontsize = 14)
plt.legend(loc = 'lower right')
plt.grid(True)

# Save the learnt trajectory for the next part
path_cbf = copy.deepcopy(x_list)
vel_cbf = copy.deepcopy(x_dot_list)

# plt.show()
# print(">> End of the script")

# =============================================================================
# CASE 6: DMPs with obstacles as CBF ON REAL ROBOT
# =============================================================================
K_approx = 0.0001  # approximation gain

# Reference trajectory (Cartesian coordinates)
N = 1000  # discretization points
a0 = 1.0  # ellipse major axis
a1 = 1.0  # ellipse minor axis
t = np.linspace(0,np.pi,N)  # time
x = a0*np.sin(t)  # x
y = a1*np.cos(t)  # y
dx = a0*np.cos(t)  # dx
dy = -a1*np.sin(t)  # dy
ref_path = np.vstack((x,y)).T  # reference path
ref_vel = np.vstack((dx,dy)).T  # reference velocity

# DMPs training
n_bfs = 100  # number of basis functions
time_step = 0.01  # time-step (default 0.01)
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 115, dt = time_step, T = t[-1],
                              alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs (K=115)
dmp_traj.imitate_path(x_des = ref_path)  # train the DMPs

# DMPs execution (no CBF)
dmp_traj.x_0 = np.array([-2., -0.5])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([-2., 1.5])  # new goal in cartesian coordinates
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
cbf = CBF()  # CBF initialization
delta_0 = 0.05  # small constant for control barrier function
eta = 0.25 # repulsive gain factor
r_min = 0.25  # radius over which the repulsive potential field is active
gamma = 100.0  # maximum acceleration for the robot
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
obstacle_centers = []  # obstacle center list
num_obstacles = 60  # number of obstacles
radius = 0.15  # radius of the obstacles REAL
for i in range(num_obstacles):
    # obstacle_centers.append(learnt_path[int(int((i+1)*len(learnt_path)/(num_obstacles+1)))])  # obstacle center SIMULATION
    obstacle_centers.append(np.array([-2.6, -0.2]) + radius * np.array([np.cos(2*np.pi*i/float(num_obstacles)), np.sin(2*np.pi*i/float(num_obstacles))]))  # obstacle center REAL

# ALPHA = 50
alpha = 50  # CBF gain
obs_force = np.array([0.,0.])  # no obstacle external force
potentials_cbf = []
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    potential = 0.
    external_force_total = np.array([0.,0.])
    for obstacle_center in obstacle_centers:
        external_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha = alpha, exp = 1.0, delta_0 = delta_0, eta = eta, r_min = r_min, gamma = gamma, obs_center = obstacle_center,
                                                      obs_force = obs_force, K_appr = K_approx, type = 'obstacle')
        external_force_total += external_force
        potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)

    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force_total + obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))
    potentials_cbf.append(potential)

# Save the learnt trajectory for the next part
path_cbf = copy.deepcopy(x_list)
vel_cbf = copy.deepcopy(x_dot_list)

# ALPHA = 10
alpha = 10  # CBF gain
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
obs_force = np.array([0.,0.])  # no obstacle external force
potentials_cbf = []
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    potential = 0.
    external_force_total = np.array([0.,0.])
    for obstacle_center in obstacle_centers:
        external_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha = alpha, exp = 1.0, delta_0 = delta_0, eta = eta, r_min = r_min, gamma = gamma, obs_center = obstacle_center,
                                                      obs_force = obs_force, K_appr = K_approx, type = 'obstacle')
        external_force_total += external_force
        potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
        
    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force_total + obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))
    potentials_cbf.append(potential)

# Save the learnt trajectory for the next part
path_cbf_10 = copy.deepcopy(x_list)
vel_cbf_10 = copy.deepcopy(x_dot_list)

dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
goal_tol = 0.01 # goal tolerance
potentials = []
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = np.array([0.,0.])
    potential = 0.
    for obstacle_center in obstacle_centers:
        obs_force += gen_dynamic_force(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
        potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
    
    x, x_dot, x_ddot = dmp_traj.step(external_force = obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))
    potentials.append(potential)

alpha = 10  # CBF gain
dmp_traj.reset_state()  # reset the state of the DMPs
x_list_10 = np.array(dmp_traj.x) # x, y
x_dot_list_10 = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list_10 = np.array(dmp_traj.ddx)  # a_x, a_y

plt.figure(6, figsize=(8, 6), tight_layout=True)
plt.plot(x_list[:,0],x_list[:,1], label = 'DMP')
plt.xlim((-3.5,1))
plt.plot(path_cbf[:,0],path_cbf[:,1], label = r'CMP, $\alpha = 50$')
plt.xlim((-3.5,1))
plt.plot(path_cbf_10[:,0],path_cbf_10[:,1], "m-", label = r'CMP, $\alpha = 10$')
plt.axis('equal')
plt.plot(learnt_path[:,0],learnt_path[:,1],'--',label = 'ref.')
plt.plot(dmp_traj.x_goal[0],dmp_traj.x_goal[1],'gx')
plt.plot(dmp_traj.x_0[0],dmp_traj.x_0[1],'bo',label = 'start')
n_o = 0
for obstacle_center in obstacle_centers:
    if n_o == 0:
        plt.plot(obstacle_center[0],obstacle_center[1],'yo',label = 'obstacle')
    else:
        plt.plot(obstacle_center[0],obstacle_center[1],'yo')
    n_o = n_o + 1
plt.xlabel('$x$ [m]', fontsize = 14)
plt.ylabel('$y$ [m]', fontsize = 14)
plt.legend(loc = 'lower right')
plt.grid(True)

# Save the learnt trajectory for the next part
path_cbf = copy.deepcopy(x_list)
vel_cbf = copy.deepcopy(x_dot_list)

plt.show()
print(">>  End of the script")