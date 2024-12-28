import numpy as np 
import copy 
import matplotlib.pyplot as plt
from matplotlib import rc
from dmp import dmp
import bezier_interp as bz
from dmp import dmp, obstacle_superquadric as obs
from cbf import CBF
from ddmr import DDMR

'''
CBFs = Control Barrier Functions
DMPs = Dynamic Movement Primitives

CBFs are used to ensure safety in the system, while DMPs are used to generate smooth trajectories.
'''

# Dynamic parameters
mu_s = 0.7  # static friction coefficient
mu_d = 0.56  # dynamic friction coefficient (80% of static friction)
omega_max = 2.83  # maximum angular velocity [rad/s]
g = 9.81 # gravity acceleration [m/s^2]
alpha = 10 # extended class-K function parameter (straight line, default 50)
exp = 1 # exponent of the extended class-K function, it must be an odd number (leave it as 1)
K_appr = 0.0001  # approximation of the constraint (default 0.0001)

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
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 68, dt = 0.01, T = t[-1],
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

# OBSTACLE (NO CBF)
dmp_traj.x_0 = np.array([3.0, 0.0])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([-2.5, 0.0])  # new goal in cartesian coordinates
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

# obstacle_center = np.array(learnt_path[int(3*len(learnt_path)/5)]) + np.array([-0.05,-0.05])  # obstacle center
obstacle_center = np.array(learnt_path[int(4*len(learnt_path)/7)]) + np.array([-0.05,-0.05])  # obstacle center
radius = 0.2
obstacle_axis = np.ones(dmp_traj.n_dmps) * radius

# Superquadric parameters for obstacle
lmbda = 2.0  # gain of relative orientation function (-lambda*cos(theta))
beta = 2.0  # exponent of the relative orientation function (-cos(theta)^beta)
eta = 1.0  # exponent of the superquadric function (C^eta(x))
obstacle = obs.Obstacle_Dynamic(center = obstacle_center, axis = obstacle_axis, 
                                lmbda = lmbda, beta=beta, eta=eta, coeffs = np.ones(dmp_traj.n_dmps))

goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = obstacle.gen_external_force(dmp_traj.x, dmp_traj.dx)
    x, x_dot, x_ddot = dmp_traj.step(external_force=obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# Save the learnt trajectory for the next part
obs_path_nocbf = copy.deepcopy(x_list)
obs_vel_nocbf = copy.deepcopy(x_dot_list)


# OBSTACLE (WITH CBF)
cbf = CBF()  # CBF initialization
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y
obstacle = obs.Obstacle_Dynamic(center = obstacle_center, axis = obstacle_axis, 
                                lmbda = lmbda, beta=beta, eta=eta, coeffs = np.ones(dmp_traj.n_dmps))

goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = obstacle.gen_external_force(dmp_traj.x, dmp_traj.dx)
    external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha, mu_s, g, exp, obs_force, omega_max, K_appr)  # compute the external force
    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force + obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# Save the learnt trajectory for the next part
obs_path_cbf = copy.deepcopy(x_list)
obs_vel_cbf = copy.deepcopy(x_dot_list)
obs_acc_cbf = copy.deepcopy(x_ddot_list)


# Path
plt.figure(1)
plt.plot(obs_path_nocbf[:,0],obs_path_nocbf[:,1],'r-',label = 'robot path (no cbf)')
plt.plot(obs_path_cbf[:,0],obs_path_cbf[:,1],'b-',label = 'robot path (with cbf)')
circle = plt.Circle(obstacle_center, radius, color='darkgreen', fill=False, linestyle='-', label='obstacle', linewidth = 2)
plt.gca().add_patch(circle)
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend()

# Velocity in inertial frame
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(obs_vel_nocbf[:,0],'r-',label = 'robot velocity (no cbf)')
plt.plot(obs_vel_cbf[:,0],'b-',label = 'robot velocity (with cbf)')
plt.ylabel(r'$\dot{x}$ [m/s]')
plt.xlabel('Time [s]')
plt.legend()

plt.subplot(2,1,2)
plt.plot(obs_vel_nocbf[:,1],'r-',label = 'robot velocity (no cbf)')
plt.plot(obs_vel_cbf[:,1],'b-',label = 'robot velocity (with cbf)')
plt.ylabel(r'$\dot{y}$ [m/s]')
plt.xlabel('Time [s]')
plt.legend()

# Forward velocity in body frame
vx_nocbf = np.sqrt(obs_vel_nocbf[:,0]**2+obs_vel_nocbf[:,1]**2)
vx_cbf = np.sqrt(obs_vel_cbf[:,0]**2+obs_vel_cbf[:,1]**2)
constraint = np.sqrt(mu_s**2*g**2-K_appr)/omega_max

plt.figure(3)
plt.plot(vx_nocbf,'r-',label = 'robot velocity (no cbf)')
plt.plot(vx_cbf,'b-',label = 'robot velocity (with cbf)')
plt.plot(constraint*np.ones(len(vx_nocbf)),'r--',label = r'$\mu_s g/\omega_{max}$')
plt.xlabel('Time [s]')
plt.ylabel(r'$v_x$ [m/s]')
plt.legend()


# ROBOT CONTROL
# no cbf
robot = DDMR()  # robot initialization at default parameters
x0 = 3.0
y0 = 0.0
theta0 = 1.657  # it has to be tuned case-by-case since the control is open loop
robot.set_state([x0, y0, theta0])  # set the robot state ([m, m, rad])
Tf = t[-1]  # final time (the same given to DMPs training)
vx_ref, omega_ref = robot.get_ref_velocity(Tf, obs_path_nocbf, obs_vel_nocbf)  # get the reference velocity

# Control loop (no cbf)
state_list = []
state_list.append(robot.state)
mode_list = []
mode_list.append(robot.mode)
for i in range(1,len(obs_path_nocbf)):
    robot.state, robot.mode = robot.dynamics_step(dt = 0.01, u = np.array([vx_ref[i], omega_ref[i]]))
    state_list.append(robot.state)
    mode_list.append(robot.mode)

robot_path_nocbf = np.array(state_list)
robot_mode_nocbf = np.array(mode_list)

# cbf
robot.set_state([x0, y0, theta0])  # set the robot state ([m, m, rad])
Tf = t[-1]  # final time (the same given to DMPs training)
vx_ref, omega_ref = robot.get_ref_velocity(Tf, obs_path_cbf, obs_vel_cbf)  # get the reference velocity

# Control loop (cbf)
state_list = []
state_list.append(robot.state)
mode_list = []
mode_list.append(robot.mode)
for i in range(1,len(obs_path_cbf)):
    robot.state, robot.mode = robot.dynamics_step(dt = 0.01, u = np.array([vx_ref[i], omega_ref[i]]))
    state_list.append(robot.state)
    mode_list.append(robot.mode)

robot_path_cbf = np.array(state_list)
robot_mode_cbf = np.array(mode_list)

plt.figure(4)
plt.plot(robot_path_nocbf[:,0],robot_path_nocbf[:,1],'r-',label = 'robot path (no cbf)')
plt.plot(robot_path_cbf[:,0],robot_path_cbf[:,1],'b-',label = 'robot path (with cbf)')
circle = plt.Circle(obstacle_center, radius, color='darkgreen', fill=False, linestyle='-', label='obstacle', linewidth = 2)
plt.gca().add_patch(circle)
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.legend()

plt.show()
print(">> End of the script")

