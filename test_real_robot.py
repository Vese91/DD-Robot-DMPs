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

# Useful functions
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

# =============================================================================
# CASE: DMPs with obstacles as CBF on real robot
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
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 45, dt = time_step, T = t[-1],
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

plt.figure(4, figsize=(8, 6), tight_layout=True)
# Main plot
plt.plot(x_list[:, 0], x_list[:, 1], label='DMP')
plt.plot(path_cbf[:, 0], path_cbf[:, 1], label=r'CMP, $\alpha = 50$')
plt.plot(path_cbf_10[:, 0], path_cbf_10[:, 1], "m-", label=r'CMP, $\alpha = 10$')
plt.axis('equal')
plt.plot(learnt_path[:, 0], learnt_path[:, 1], '--', label='ref.')
plt.plot(dmp_traj.x_goal[0], dmp_traj.x_goal[1], 'gs', label='goal')
plt.plot(dmp_traj.x_0[0], dmp_traj.x_0[1], 'bD', label='start')
n_o = 0
for obstacle_center in obstacle_centers:
    if n_o == 0:
        plt.plot(obstacle_center[0], obstacle_center[1], 'yo', label='obstacle')
    else:
        plt.plot(obstacle_center[0], obstacle_center[1], 'yo')
    n_o = n_o + 1
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.xlim([-3.25, -0.50])
plt.ylim([-0.60, 1.60])
plt.legend(loc='lower right')
plt.grid(True)

# Nested plot
inset_ax = plt.axes([0.57, 0.50, 0.4, 0.4])  # [x, y, width, height] in normalized coordinates
inset_ax.plot(x_list[:, 0], x_list[:, 1], label='DMP')
inset_ax.plot(path_cbf[:, 0], path_cbf[:, 1], label=r'CMP, $\alpha = 50$')
inset_ax.plot(path_cbf_10[:, 0], path_cbf_10[:, 1], "m-", label=r'CMP, $\alpha = 10$')
inset_ax.plot(learnt_path[:, 0], learnt_path[:, 1], '--', label='ref.')
inset_ax.plot(dmp_traj.x_goal[0], dmp_traj.x_goal[1], 'gs', label='goal')
inset_ax.plot(dmp_traj.x_0[0], dmp_traj.x_0[1], 'bD', label='start')
for obstacle_center in obstacle_centers:
    inset_ax.plot(obstacle_center[0], obstacle_center[1], 'yo')
inset_ax.set_xlim([-3.00, -2.10])  # Adjust limits to focus on a specific area
inset_ax.set_ylim([-0.55, 0.15])
inset_ax.grid(True)

plt.show()
print(">>  End of the script")