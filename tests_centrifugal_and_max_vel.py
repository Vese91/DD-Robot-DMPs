import numpy as np
import matplotlib.pyplot as plt
import copy 
import matplotlib.pyplot as plt
from matplotlib import rc
from dmp import dmp
#import bezier_interp as bz
from dmp import dmp, obstacle_superquadric as obs
from cbf import CBF
#from ddmr import DDMR
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
plt.title("Constraint on maximum velocity", fontsize=16)
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

plt.figure(figsize=(8, 6), tight_layout=True)
plt.title("Constraint on centrifugal acceleration", fontsize=16)
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
plt.show()
