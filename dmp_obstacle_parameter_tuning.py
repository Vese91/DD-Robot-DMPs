import numpy as np
import matplotlib.pyplot as plt
import copy 
import matplotlib.pyplot as plt
from matplotlib import rc
from dmp import dmp
from cbf import CBF
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
# CASE 4: DMPs with obstacles as CBF (narrow passage)
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
r_min = 0.25  # radius over which the repulsive potential field is active
obstacle_centers = []  # obstacle center list
num_obstacles = 10  # number of obstacles
radius = 0.01
# Place obstacles on one side of the corridor
for i in range(int(np.round(num_obstacles / 2))):
    index = min(int((i + 1) * len(learnt_path) / 6), len(learnt_path) - 1)  # Ensure index is within bounds
    obstacle_centers.append(learnt_path[index] + np.array([0, 0.05]))  # offset upward

# Place obstacles on the other side of the corridor
for i in range(int(np.round(num_obstacles / 2))):
    index = min(int((i + 1) * len(learnt_path) / 6), len(learnt_path) - 1)  # Ensure index is within bounds
    obstacle_centers.append(learnt_path[index] + np.array([0, -0.05]))  # offset downward

# Add two moving obstacles
moving_obstacles = [
    {"pos": learnt_path[30:50], "radius": 0.05, "vel": learnt_vel[30:50]},
    {"pos": learnt_path[-50:-30], "radius": 0.05, "vel": learnt_vel[-50:-30]},
]

for gamma in [90, 120]:
    for eta in [0.01, 0.071]:
        dmp_traj.reset_state()  # reset the state of the DMPs
        x_list = np.array(dmp_traj.x)  # x, y
        x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
        x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
        goal_tol = 0.01  # goal tolerance
        obs_force = np.array([0., 0.])  # no obstacle external force
        potentials_cbf = []
        t = 0  # index for moving obstacles path and vel
        invert_order = False
        Ts = 0.01  # time step
        while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
            potential = 0.
            external_force_total = np.array([0., 0.])
            for obstacle_center in obstacle_centers:
                external_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha=alpha, exp=1.0, delta_0=delta_0, eta=eta, r_min=r_min, gamma=gamma, obs_center=obstacle_center,
                                                                obs_force=obs_force, K_appr=K_approx, type='obstacle')
                external_force_total += external_force
                potential = gen_potential(gamma=gamma, eta=eta, v_ego=dmp_traj.dx, v_obs=np.array([0, 0]), p_obs=obstacle_center, p_ego=dmp_traj.x)

            # Update moving obstacles
            for moving_obstacle in moving_obstacles:
                if t < len(moving_obstacle["pos"]) and t >= 0:
                    obs_center = moving_obstacle["pos"][t]
                    if invert_order:
                        obs_vel = -moving_obstacle["vel"][t]
                    else:
                        obs_vel = moving_obstacle["vel"][t]
                    if invert_order:
                        t = t-1
                    else:
                        t = t+1
                else:
                    invert_order = not invert_order
                    if t < 0:
                        t = 0
                    else:
                        t = len(moving_obstacle["pos"]) - 1
                # print(obs_center)
                # print(obs_vel)
                # print("=================")
                external_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha=alpha, exp=1.0, delta_0=delta_0, eta=eta, r_min=r_min, gamma=gamma, obs_center=obs_center,
                                                                obs_force=obs_force, K_appr=K_approx, type='obstacle')
                external_force_total += external_force

            x, x_dot, x_ddot = dmp_traj.step(external_force=external_force_total + obs_force)  # execute the DMPs
            x_list = np.vstack((x_list, x))
            x_dot_list = np.vstack((x_dot_list, x_dot))
            x_ddot_list = np.vstack((x_ddot_list, x_ddot))
            potentials_cbf.append(potential)
            # t += Ts  # increment time

        # Save the learnt trajectory for the next part
        path_cbf = copy.deepcopy(x_list)
        vel_cbf = copy.deepcopy(x_dot_list)

        plt.figure(4, figsize=(8, 6), tight_layout=True)
        plt.subplots_adjust(hspace=0.3)  # Adjust the space between the subplots

        dmp_traj.reset_state()  # reset the state of the DMPs
        x_list = np.array(dmp_traj.x)  # x, y
        x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
        x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

        goal_tol = 0.01  # goal tolerance
        potentials = []
        t = 0
        invert_order = False
        while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
            obs_force = np.array([0., 0.])
            potential = 0.
            for obstacle_center in obstacle_centers:
                obs_force += gen_dynamic_force(gamma=gamma, eta=eta, v_ego=dmp_traj.dx, v_obs=np.array([0, 0]), p_obs=obstacle_center, p_ego=dmp_traj.x)
                potential = gen_potential(gamma=gamma, eta=eta, v_ego=dmp_traj.dx, v_obs=np.array([0, 0]), p_obs=obstacle_center, p_ego=dmp_traj.x)

            # Update moving obstacles
            for moving_obstacle in moving_obstacles:
                if t < len(moving_obstacle["pos"]) and t >= 0:
                    obs_center = moving_obstacle["pos"][t]
                    if invert_order:
                        obs_vel = -moving_obstacle["vel"][t]
                    else:
                        obs_vel = moving_obstacle["vel"][t]
                    if invert_order:
                        t = t-1
                    else:
                        t = t+1
                else:
                    invert_order = not invert_order
                    if t < 0:
                        t = 0
                    else:
                        t = len(moving_obstacle["pos"]) - 1
                obs_force += gen_dynamic_force(gamma=gamma, eta=eta, v_ego=dmp_traj.dx, v_obs=obs_vel, p_obs=obs_center, p_ego=dmp_traj.x)

            x, x_dot, x_ddot = dmp_traj.step(external_force=obs_force)  # execute the DMPs
            x_list = np.vstack((x_list, x))
            x_dot_list = np.vstack((x_dot_list, x_dot))
            x_ddot_list = np.vstack((x_ddot_list, x_ddot))
            potentials.append(potential)
            # t += Ts  # increment time

        plt.plot(x_list[:, 0], x_list[:, 1], 'r-', label='DMP')
        plt.plot(path_cbf[:, 0], path_cbf[:, 1], 'b-', label='CMP')
        plt.plot(learnt_path[:, 0], learnt_path[:, 1], '--', label='ref.')
        plt.plot(dmp_traj.x_goal[0], dmp_traj.x_goal[1], 'gs')
        plt.plot(dmp_traj.x_0[0], dmp_traj.x_0[1], 'bD', label='start')
        n_o = 0
        for obstacle_center in obstacle_centers:
            if n_o == 0:
                plt.plot(obstacle_center[0], obstacle_center[1], 'ko', label='obstacle')
            else:
                plt.plot(obstacle_center[0], obstacle_center[1], 'ko')
            n_o = n_o + 1

        # Plot moving obstacles and their trajectories
        for i, moving_obstacle in enumerate(moving_obstacles):
            trajectory = np.array(moving_obstacle["pos"])
            if i == 0:  # Label only the first moving obstacle
                plt.plot(trajectory[:, 0], trajectory[:, 1], 'k--', linewidth=4, label='moving obstacle')
            else:
                plt.plot(trajectory[:, 0], trajectory[:, 1], 'k--', linewidth=4)

        # plt.title(f'$\gamma$={gamma}, $\eta$={eta:.2f}', fontsize=32)
        plt.xlabel('$x$ [m]', fontsize=32)
        plt.ylabel('$y$ [m]', fontsize=32)
        plt.tick_params(axis='both', which='major', labelsize=32)
        plt.legend(loc='lower center', fontsize=32)
        plt.grid(True)

        # Save trajectory without CBF
        path_nocbf = copy.deepcopy(x_list)
        vel_nocbf = copy.deepcopy(x_dot_list)

        # Create figure(5) with a nested plot
        plt.figure(5, figsize=(8, 6), tight_layout=True)
        plt.subplots_adjust(hspace=0.3)

        # Main plot
        plt.plot(x_list[:, 0], x_list[:, 1], 'r-', label='DMP')
        plt.plot(path_cbf[:, 0], path_cbf[:, 1], 'b-', label='CMP')
        plt.plot(learnt_path[:, 0], learnt_path[:, 1], '--', label='ref.')
        plt.plot(dmp_traj.x_goal[0], dmp_traj.x_goal[1], 'gs', label='goal')
        plt.plot(dmp_traj.x_0[0], dmp_traj.x_0[1], 'bD', label='start')
        n_o = 0
        for obstacle_center in obstacle_centers:
            if n_o == 0:
                plt.plot(obstacle_center[0], obstacle_center[1], 'ko', label='obstacle')
            else:
                plt.plot(obstacle_center[0], obstacle_center[1], 'ko')
            n_o = n_o + 1

        # Plot moving obstacles and their trajectories
        for i, moving_obstacle in enumerate(moving_obstacles):
            trajectory = np.array(moving_obstacle["pos"])
            if i == 0:  # Label only the first moving obstacle
                plt.plot(trajectory[:, 0], trajectory[:, 1], 'k--', linewidth=4, label='moving obstacle')
            else:
                plt.plot(trajectory[:, 0], trajectory[:, 1], 'k--', linewidth=4)

        plt.xlabel('$x$ [m]', fontsize=14)
        plt.ylabel('$y$ [m]', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True)

        # Nested plot
        nested_ax = plt.axes([0.32, 0.15, 0.4, 0.4])  # Adjust position and size

        # Plot x_list in darker red scale
        for i in range(len(x_list)-1):
            nested_ax.plot(x_list[i:i+2, 0], x_list[i:i+2, 1], color=plt.cm.Reds(0.1 + 0.9*(i / len(x_list))))
        # nested_ax.plot(x_list[:, 0], x_list[:, 1], 'r-', label='DMP')

        # Plot path_cbf in darker blue scale
        for i in range(len(path_cbf) - 1):
            nested_ax.plot(path_cbf[i:i+2, 0], path_cbf[i:i+2, 1], color=plt.cm.Blues(0.1 + 0.9*(i / len(path_cbf))))
        # nested_ax.plot(path_cbf[:, 0], path_cbf[:, 1], 'b-', label='CMP')

        nested_ax.plot(learnt_path[:, 0], learnt_path[:, 1], '--', label='ref.')
        nested_ax.plot(dmp_traj.x_goal[0], dmp_traj.x_goal[1], 'gs')
        nested_ax.plot(dmp_traj.x_0[0], dmp_traj.x_0[1], 'bD', label='start')
        for obstacle_center in obstacle_centers:
            nested_ax.plot(obstacle_center[0], obstacle_center[1], 'ko')
        for moving_obstacle in moving_obstacles:
            trajectory = np.array(moving_obstacle["pos"])
            for i in range(len(trajectory) - 1):
                nested_ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1],':', color=plt.cm.Greys(0.2 + 0.9*(i / len(trajectory))), linewidth=4)
        
        nested_ax.set_xlim([-3, -1.25])  # Example zoomed-in region
        nested_ax.set_ylim([0.2, 0.55])
        nested_ax.grid(True)

        # Save the learnt trajectory for the next part
        # path_cbf = copy.deepcopy(x_list)
        # vel_cbf = copy.deepcopy(x_dot_list)

        # Infinity norm of trajectories
        # Function to downsample a path to a target number of points
        def downsample_path(path, target_len):
            idx = np.linspace(0, len(path) - 1, target_len).astype(int)
            return path[idx]

        # Downsample path_cbf and path_nocbf to match the reference path's length
        path_cbf_ds = downsample_path(path_cbf, len(learnt_path))
        path_nocbf_ds = downsample_path(path_nocbf, len(learnt_path))

        # Squared errors
        sq_err_cbf = np.sum((learnt_path - path_cbf_ds) ** 2, axis=1)
        sq_err_nocbf = np.sum((learnt_path - path_nocbf_ds) ** 2, axis=1)

        # RMSE and standard deviation for CBF path
        rmse_cbf = np.sqrt(np.mean(sq_err_cbf))
        std_cbf = np.std(np.sqrt(sq_err_cbf))

        # RMSE and standard deviation for no CBF path
        rmse_nocbf = np.sqrt(np.mean(sq_err_nocbf))
        std_nocbf = np.std(np.sqrt(sq_err_nocbf))

        print("gamma:", gamma, "eta:", eta)
        print("RMSE of CMPs: ", rmse_cbf, "Std:", std_cbf)
        print("RMSE of DMPs: ", rmse_nocbf, "Std:", std_nocbf)
        print("max of DMPs", np.max(np.sqrt(sq_err_nocbf)))
        print("max of CMPs", np.max(np.sqrt(sq_err_cbf)))

        # Bar plot for RMSE and Std
        labels = ['CMP', 'DMP']
        rmse_values = [rmse_cbf, rmse_nocbf]
        std_values = [std_cbf, std_nocbf]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax1 = plt.subplots(figsize=(8, 3))
        # Use a lighter blue for CMPs
        bar_colors = ['#7ec8e3', 'orange']  # '#7ec8e3' is a light blue
        center_shift = 0.005  # even smaller shift, more centered

        # Plot left bar (CMPs) on left y-axis, shifted right
        bar1 = ax1.bar(
            x[0] + center_shift, rmse_values[0], width, yerr=std_values[0], capsize=8,
            color=bar_colors[0], label=labels[0],
            edgecolor='black', linewidth=2,
            error_kw=dict(lw=3)
        )
        ax1.set_ylabel('CMP error scale', color='black', fontsize=32)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.tick_params(axis='x', labelsize=32)

        # Plot right bar (DMPs) on right y-axis, shifted left
        ax2 = ax1.twinx()
        bar2 = ax2.bar(
            x[1] - center_shift, rmse_values[1], width, yerr=std_values[1], capsize=8,
            color=bar_colors[1], label=labels[1],
            edgecolor='black', linewidth=2,
            error_kw=dict(lw=3)
        )
        ax2.set_ylabel('DMP error scale', color='black', fontsize=32)
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='x', labelsize=32)

        # Move x-axis labels to center between bars
        ax1.set_xticks([x[0] + center_shift/2, x[1] - center_shift/2])
        ax1.set_xticklabels(labels, fontsize=32)

        # Add background grid
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax2.grid(False)

        fig.tight_layout()
        plt.show()

        print(">> End of first part")


# # =============================================================================
# # CASE 5: DMPs with obstacles as CBF (moving obstacle)
# # =============================================================================
# K_approx = 0.0001  # approximation gain

# # Reference trajectory (Cartesian coordinates)
# N = 1000  # discretization points
# a0 = 3.0  # ellipse major axis
# a1 = 1.0  # ellipse minor axis
# t = np.linspace(0, np.pi, N)  # time
# x = a0 * np.cos(t)  # x
# y = a1 * np.sin(t)  # y
# dx = -a0 * np.sin(t)  # dx
# dy = a1 * np.cos(t)  # dy
# ref_path = np.vstack((x, y)).T  # reference path
# ref_vel = np.vstack((dx, dy)).T  # reference velocity

# # DMPs training
# n_bfs = 100  # number of basis functions
# time_step = 0.01  # time-step
# dmp_traj = dmp.DMPs_cartesian(n_dmps=2, n_bfs=n_bfs, K=100, dt=time_step, T=t[-1],
#                               alpha_s=2.0, tol=3.0 / 100, rescale="rotodilatation", basis="gaussian")  # set up the DMPs (K=115)
# dmp_traj.imitate_path(x_des=ref_path)  # train the DMPs

# # DMPs execution (no CBF)
# dmp_traj.x_0 = np.array([3.0, 0.0])  # new start in cartesian coordinates
# dmp_traj.x_goal = np.array([-2.5, 0.0])  # new goal in cartesian coordinates
# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list = np.array(dmp_traj.x)  # x, y
# x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# # Loop
# goal_tol = 0.01  # goal tolerance
# while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
#     x, x_dot, x_ddot = dmp_traj.step()  # execute the DMPs
#     x_list = np.vstack((x_list, x))
#     x_dot_list = np.vstack((x_dot_list, x_dot))
#     x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# # Save the learnt trajectory for the next part
# learnt_path = copy.deepcopy(x_list)
# learnt_vel = copy.deepcopy(x_dot_list)

# # DMPs with Obstacle as CBF
# alpha = 50  # CBF gain
# cbf = CBF()  # CBF initialization
# delta_0 = 0.05  # small constant for control barrier function
# eta = 0.05  # repulsive gain factor (default 0.05)
# r_min = 0.25  # radius over which the repulsive potential field is active
# gamma = 120.0  # maximum acceleration for the robot (default 100.0)
# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list = np.array(dmp_traj.x)  # x, y
# x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# obstacle_centers = []  # obstacle center list
# num_obstacles = 10  # number of obstacles
# num_obst_points = 1  # number of points to define the obstacle
# radius = 0.0
# for i in range(num_obstacles):
#     for j in range(num_obst_points):
#         obstacle_centers.append(learnt_path[int(int((i+1)*len(learnt_path)/(num_obstacles+1)))] + radius * np.array([np.cos(2*np.pi*j/float(num_obst_points)), np.sin(2*np.pi*j/float(num_obst_points))]))  # obstacle center

# goal_tol = 0.01 # goal tolerance
# obs_force = np.array([0.,0.])  # no obstacle external force
# potentials_cbf = []
# A = 0.05 # amplitude of the moving obstacle (default 0.05)
# omega = np.pi/2 # frequency of the moving obstacle
# t = 0.0  # initial time
# Ts = 0.01  # time step for the moving obstacle
# while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
#     potential = 0.
#     external_force_total = np.array([0.,0.])
#     for obstacle_center in obstacle_centers:
#         external_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha = alpha, exp = 1.0, delta_0 = delta_0, eta = eta, r_min = r_min, gamma = gamma, obs_center = obstacle_center,
#                                                       obs_force = obs_force, K_appr = K_approx, type = 'obstacle')
#         external_force_total += external_force
#         potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)

#         # Update the obstacle center position based on the sine wave
#         obstacle_center[0] += Ts*A*np.sin(omega*t)  # x position of the moving obstacle
#         obstacle_center[1] += Ts*A*np.cos(omega*t)  # y position of the moving obstacle
        
#     x, x_dot, x_ddot = dmp_traj.step(external_force = external_force_total + obs_force)  # execute the DMPs
#     x_list = np.vstack((x_list, x))
#     x_dot_list = np.vstack((x_dot_list, x_dot))
#     x_ddot_list = np.vstack((x_ddot_list, x_ddot))
#     potentials_cbf.append(potential)
#     t += Ts  # increment time

# # Save the learnt trajectory for the next part
# path_cbf = copy.deepcopy(x_list)
# vel_cbf = copy.deepcopy(x_dot_list)

# plt.figure(5, figsize=(8, 3), tight_layout=True)
# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list = np.array(dmp_traj.x) # x, y
# x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

# # reset the obstacle centers to the original position
# obstacle_centers = []  # obstacle center list
# num_obstacles = 10  # number of obstacles
# num_obst_points = 1  # number of points to define the obstacle
# for i in range(num_obstacles):
#     for j in range(num_obst_points):
#         obstacle_centers.append(learnt_path[int(int((i+1)*len(learnt_path)/(num_obstacles+1)))] + radius * np.array([np.cos(2*np.pi*j/float(num_obst_points)), np.sin(2*np.pi*j/float(num_obst_points))]))  # obstacle center

# goal_tol = 0.01 # goal tolerance
# potentials = []
# step = 0
# t = 0.0 # reset time
# while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
#     obs_force = np.array([0.,0.])
#     potential = 0.
#     for obstacle_center in obstacle_centers:
#         obs_force += gen_dynamic_force(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
#         potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)

#         # Update the obstacle center position based on the sine wave
#         obstacle_center[0] += Ts*A*np.sin(omega*t)  # x position of the moving obstacle
#         obstacle_center[1] += Ts*A*np.cos(omega*t)  # y position of the moving obstacle
#         plt.plot(obstacle_center[0], obstacle_center[1], 'o', color='gray')  # plot in grayscale
        
#     x, x_dot, x_ddot = dmp_traj.step(external_force = obs_force)  # execute the DMPs
#     step = step + 1
#     print(step)
#     x_list = np.vstack((x_list, x))
#     x_dot_list = np.vstack((x_dot_list, x_dot))
#     x_ddot_list = np.vstack((x_ddot_list, x_ddot))
#     potentials.append(potential)
#     t += Ts  # increment time

# plt.subplots_adjust(hspace=0.3)  # Adjust the space between the subplots
# plt.plot(x_list[:,0],x_list[:,1],'r-',label = 'DMP')
# plt.plot(path_cbf[:,0],path_cbf[:,1],'b-',label = 'CMP')
# plt.plot(learnt_path[:,0],learnt_path[:,1],'--',label = 'ref.')
# plt.plot(dmp_traj.x_goal[0],dmp_traj.x_goal[1],'gs')
# plt.plot(dmp_traj.x_0[0],dmp_traj.x_0[1],'bD',label = 'start')
# n_o = 0
# for obstacle_center in obstacle_centers:
#     if n_o == 0:
#         plt.plot(obstacle_center[0], obstacle_center[1], 'o', color='black', label='obstacle')
#     else:
#         plt.plot(obstacle_center[0], obstacle_center[1], 'o', color='black')
#     n_o = n_o + 1
# plt.xlabel('$x$ [m]', fontsize = 14)
# plt.ylabel('$y$ [m]', fontsize = 14)
# plt.legend(loc = 'upper right')
# plt.grid(True)

# # Save the learnt trajectory for the next part
# path_cbf = copy.deepcopy(x_list)
# vel_cbf = copy.deepcopy(x_dot_list)

# plt.show()
# print(">> End of second part")

# # =============================================================================
# # CASE: DMPs with obstacles as CBF on real robot
# # =============================================================================
# K_approx = 0.0001  # approximation gain

# # Reference trajectory (Cartesian coordinates)
# N = 1000  # discretization points
# a0 = 1.0  # ellipse major axis
# a1 = 1.0  # ellipse minor axis
# t = np.linspace(0,np.pi,N)  # time
# x = a0*np.sin(t)  # x
# y = a1*np.cos(t)  # y
# dx = a0*np.cos(t)  # dx
# dy = -a1*np.sin(t)  # dy
# ref_path = np.vstack((x,y)).T  # reference path
# ref_vel = np.vstack((dx,dy)).T  # reference velocity

# # DMPs training
# n_bfs = 100  # number of basis functions
# time_step = 0.01  # time-step (default 0.01)
# dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 115, dt = time_step, T = t[-1],
#                               alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs (K=115)
# dmp_traj.imitate_path(x_des = ref_path)  # train the DMPs

# # DMPs execution (no CBF)
# dmp_traj.x_0 = np.array([-2., -0.5])  # new start in cartesian coordinates
# dmp_traj.x_goal = np.array([-2., 1.5])  # new goal in cartesian coordinates
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

# # DMPs with Obstacle as CBF
# cbf = CBF()  # CBF initialization
# delta_0 = 0.05  # small constant for control barrier function
# eta = 0.25 # repulsive gain factor
# r_min = 0.25  # radius over which the repulsive potential field is active
# gamma = 100.0  # maximum acceleration for the robot
# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list = np.array(dmp_traj.x) # x, y
# x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# obstacle_centers = []  # obstacle center list
# num_obstacles = 60  # number of obstacles
# radius = 0.15  # radius of the obstacles REAL
# for i in range(num_obstacles):
#     obstacle_centers.append(np.array([-2.6, -0.2]) + radius * np.array([np.cos(2*np.pi*i/float(num_obstacles)), np.sin(2*np.pi*i/float(num_obstacles))]))  # obstacle center REAL

# # ALPHA = 50
# alpha = 50  # CBF gain
# obs_force = np.array([0.,0.])  # no obstacle external force
# potentials_cbf = []
# while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
#     potential = 0.
#     external_force_total = np.array([0.,0.])
#     for obstacle_center in obstacle_centers:
#         external_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha = alpha, exp = 1.0, delta_0 = delta_0, eta = eta, r_min = r_min, gamma = gamma, obs_center = obstacle_center,
#                                                       obs_force = obs_force, K_appr = K_approx, type = 'obstacle')
#         external_force_total += external_force
#         potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)

#     x, x_dot, x_ddot = dmp_traj.step(external_force = external_force_total + obs_force)  # execute the DMPs
#     x_list = np.vstack((x_list, x))
#     x_dot_list = np.vstack((x_dot_list, x_dot))
#     x_ddot_list = np.vstack((x_ddot_list, x_ddot))
#     potentials_cbf.append(potential)

# # Save the learnt trajectory for the next part
# path_cbf = copy.deepcopy(x_list)
# vel_cbf = copy.deepcopy(x_dot_list)

# # ALPHA = 10
# alpha = 10  # CBF gain
# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list = np.array(dmp_traj.x) # x, y
# x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# obs_force = np.array([0.,0.])  # no obstacle external force
# potentials_cbf = []
# while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
#     potential = 0.
#     external_force_total = np.array([0.,0.])
#     for obstacle_center in obstacle_centers:
#         external_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha = alpha, exp = 1.0, delta_0 = delta_0, eta = eta, r_min = r_min, gamma = gamma, obs_center = obstacle_center,
#                                                       obs_force = obs_force, K_appr = K_approx, type = 'obstacle')
#         external_force_total += external_force
#         potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
        
#     x, x_dot, x_ddot = dmp_traj.step(external_force = external_force_total + obs_force)  # execute the DMPs
#     x_list = np.vstack((x_list, x))
#     x_dot_list = np.vstack((x_dot_list, x_dot))
#     x_ddot_list = np.vstack((x_ddot_list, x_ddot))
#     potentials_cbf.append(potential)

# # Save the learnt trajectory for the next part
# path_cbf_10 = copy.deepcopy(x_list)
# vel_cbf_10 = copy.deepcopy(x_dot_list)

# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list = np.array(dmp_traj.x) # x, y
# x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
# goal_tol = 0.01 # goal tolerance
# potentials = []
# while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
#     obs_force = np.array([0.,0.])
#     potential = 0.
#     for obstacle_center in obstacle_centers:
#         obs_force += gen_dynamic_force(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
#         potential = gen_potential(gamma=gamma, eta=eta, v_ego = dmp_traj.dx, v_obs = np.array([0,0]), p_obs = obstacle_center, p_ego = dmp_traj.x)
    
#     x, x_dot, x_ddot = dmp_traj.step(external_force = obs_force)  # execute the DMPs
#     x_list = np.vstack((x_list, x))
#     x_dot_list = np.vstack((x_dot_list, x_dot))
#     x_ddot_list = np.vstack((x_ddot_list, x_ddot))
#     potentials.append(potential)

# alpha = 10  # CBF gain
# dmp_traj.reset_state()  # reset the state of the DMPs
# x_list_10 = np.array(dmp_traj.x) # x, y
# x_dot_list_10 = np.array(dmp_traj.dx)  # v_x, v_y
# x_ddot_list_10 = np.array(dmp_traj.ddx)  # a_x, a_y

# plt.figure(6, figsize=(8, 6), tight_layout=True)
# plt.plot(x_list[:,0],x_list[:,1], label = 'DMP')
# plt.xlim((-3.5,1))
# plt.plot(path_cbf[:,0],path_cbf[:,1], label = r'CMP, $\alpha = 50$')
# plt.xlim((-3.5,1))
# plt.plot(path_cbf_10[:,0],path_cbf_10[:,1], "m-", label = r'CMP, $\alpha = 10$')
# plt.axis('equal')
# plt.plot(learnt_path[:,0],learnt_path[:,1],'--',label = 'ref.')
# plt.plot(dmp_traj.x_goal[0],dmp_traj.x_goal[1],'gs',label = 'goal')
# plt.plot(dmp_traj.x_0[0],dmp_traj.x_0[1],'bD',label = 'start')
# n_o = 0
# for obstacle_center in obstacle_centers:
#     if n_o == 0:
#         plt.plot(obstacle_center[0],obstacle_center[1],'yo',label = 'obstacle')
#     else:
#         plt.plot(obstacle_center[0],obstacle_center[1],'yo')
#     n_o = n_o + 1
# plt.xlabel('$x$ [m]')
# plt.ylabel('$y$ [m]')
# plt.legend(loc = 'lower right')
# plt.grid(True)

# # UNDERSAMPLING
# def undersample(data, factor):
#     return data[::factor]

# # Time step ratio
# undersample_factor = int(0.1 / time_step)

# # Undersample the paths and velocities
# path_cbf = undersample(path_cbf, undersample_factor)
# vel_cbf = undersample(vel_cbf, undersample_factor)
# path_cbf_10 = undersample(path_cbf_10, undersample_factor)
# vel_cbf_10 = undersample(vel_cbf_10, undersample_factor)

# plt.figure(7, figsize=(8, 6), tight_layout=True)
# plt.plot(x_list[:,0],x_list[:,1], label = 'DMP')
# plt.xlim((-3.5,1))
# plt.plot(path_cbf[:,0],path_cbf[:,1], label = r'CMP, $\alpha = 50$')
# plt.xlim((-3.5,1))
# plt.plot(path_cbf_10[:,0],path_cbf_10[:,1], "m-", label = r'CMP, $\alpha = 10$')
# plt.axis('equal')
# plt.plot(learnt_path[:,0],learnt_path[:,1],'--',label = 'ref.')
# plt.plot(dmp_traj.x_goal[0],dmp_traj.x_goal[1],'gs',label = 'goal')
# plt.plot(dmp_traj.x_0[0],dmp_traj.x_0[1],'bD',label = 'start')
# n_o = 0
# for obstacle_center in obstacle_centers:
#     if n_o == 0:
#         plt.plot(obstacle_center[0],obstacle_center[1],'yo',label = 'obstacle')
#     else:
#         plt.plot(obstacle_center[0],obstacle_center[1],'yo')
#     n_o = n_o + 1
# plt.xlabel('$x$ [m]')
# plt.ylabel('$y$ [m]')
# plt.legend(loc = 'lower right')
# plt.grid(True)

# # Linear and angular velocities for the robot
# sample_time = 0.1  # time step for the robot
# v = np.linalg.norm(vel_cbf, axis=1)  # linear velocity
# theta = np.arctan2(vel_cbf[:, 1], vel_cbf[:, 0])  # initial angle
# theta = np.unwrap(theta)  # unwrap the angle
# omega = np.gradient(theta,sample_time) # angular velocity
# x = path_cbf[0, 0]  # initial x position
# y = path_cbf[0, 1]  # initial y position
# robot_path = []  # robot path
# theta_0 = theta[0]  # initial angle
# for j in range(len(v)):
#     x = x + sample_time*v[j]*np.cos(theta_0)  # update x position
#     y = y + sample_time*v[j]*np.sin(theta_0)  # update y position
#     theta_0 = theta_0 + sample_time*omega[j]  # update angle
#     robot_path.append([x, y])  # append new position to the robot path

# robot_path = np.array(robot_path)  # convert to numpy array

# plt.figure(8, figsize=(8, 6), tight_layout=True)
# plt.plot(x_list[:,0],x_list[:,1], label = 'DMP')
# plt.xlim((-3.5,1))
# plt.plot(robot_path[:,0],robot_path[:,1], label = r'robot path, $\alpha = 50$')
# plt.xlim((-3.5,1))
# plt.plot(path_cbf_10[:,0],path_cbf_10[:,1], "m-", label = r'CMP, $\alpha = 10$')
# plt.axis('equal')
# plt.plot(learnt_path[:,0],learnt_path[:,1],'--',label = 'ref.')
# plt.plot(dmp_traj.x_goal[0],dmp_traj.x_goal[1],'gs',label = 'goal')
# plt.plot(dmp_traj.x_0[0],dmp_traj.x_0[1],'bD',label = 'start')
# n_o = 0
# for obstacle_center in obstacle_centers:
#     if n_o == 0:
#         plt.plot(obstacle_center[0],obstacle_center[1],'yo',label = 'obstacle')
#     else:
#         plt.plot(obstacle_center[0],obstacle_center[1],'yo')
#     n_o = n_o + 1
# plt.xlabel('$x$ [m]')
# plt.ylabel('$y$ [m]')
# plt.legend(loc = 'lower right')
# plt.grid(True)
# plt.show()


# print(">>  End of the script")