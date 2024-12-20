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
g = 9.81 # gravity acceleration [m/s^2]
alpha = 50 # extended class-K function parameter (straight line)
exp = 1 # exponent of the extended class-K function, it must be an odd number (leave it as 1)

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

# plt.plot(ref_path[:,0], ref_path[:,1],'b-',label='Reference trajectory')
# plt.plot(ref_path[0,0], ref_path[0,1],'ko',label='Start')
# plt.plot(ref_path[-1,0], ref_path[-1,1],'kx',label='Goal')
# plt.title('Ref. training trajectory')
# plt.legend()
# plt.show()

# DMPs training
n_bfs = 100  # number of basis functions
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 115, dt = 0.01, T = t[-1],
                              alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs
dmp_traj.imitate_path(x_des=ref_path)  # train the DMPs

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

# Plot the result
# plt.plot(learnt_path[:,0], learnt_path[:,1],'r--',label='reference traj.')
# plt.plot(x_list[:,0], x_list[:,1],'b-',label='executed traj. (no cbf)')
# plt.plot(x_list[0,0], x_list[0,1],'go',label='start')
# plt.plot(x_list[-1,0], x_list[-1,1],'rx',label='goal')
# circle = plt.Circle(obstacle_center, obstacle_axis[0], color='g', fill=False)  # plot a circle for the obstacle
# plt.gca().add_artist(circle)
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.legend()
# plt.subplots_adjust(left=0.086, right=0.99, top=0.99)
# plt.show()

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
    external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha, mu_s, g, exp, obs_force)
    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force + obs_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

# Plot the result
# plt.plot(learnt_path[:,0], learnt_path[:,1],'r--',label='reference traj.')
# plt.plot(x_list[:,0], x_list[:,1],'b-',label='executed traj. (with cbf)')
# plt.plot(x_list[0,0], x_list[0,1],'go',label='start')
# plt.plot(x_list[-1,0], x_list[-1,1],'rx',label='goal')
# circle = plt.Circle(obstacle_center, obstacle_axis[0], color='g', fill=False)  # plot a circle for the obstacle
# plt.gca().add_artist(circle)
# plt.xlabel('x [m]')
# plt.ylabel('y [m]')
# plt.legend()
# plt.subplots_adjust(left=0.086, right=0.99, top=0.99)
# plt.show()

# Save the learnt trajectory for the next part
obs_path_cbf = copy.deepcopy(x_list)
obs_vel_cbf = copy.deepcopy(x_dot_list)

# Centrifugal force
tVec = np.linspace(0,t[-1],len(obs_path_nocbf))
F_nocbf = (obs_path_nocbf[:,0]*obs_vel_nocbf[:,1]-obs_path_nocbf[:,1]*obs_vel_nocbf[:,0])**2 / (obs_path_nocbf[:,0]**2+obs_path_nocbf[:,1]**2)**(3/2)  # (x*dy-y*dx)^2/(x^2+y^2)^(3/2)
F_cbf = (obs_path_cbf[:,0]*obs_vel_cbf[:,1]-obs_path_cbf[:,1]*obs_vel_cbf[:,0])**2 / (obs_path_cbf[:,0]**2+obs_path_cbf[:,1]**2)**(3/2)

# Get the reference inputs for the DDMR
vx_ref, omega_ref = DDMR.get_ddmr_refinputs(tVec, obs_path_nocbf, obs_vel_nocbf)

plt.figure()
plt.subplot(2,1,1)
plt.plot(tVec,vx_ref,'b',linestyle = '-',label = r'$vx_{ref}$ (no cbf)')
plt.legend()

plt.subplot(2,1,2)
plt.plot(tVec,omega_ref,'b',linestyle = '-',label = r'$\omega_{ref}$ (no cbf)')
plt.plot()
plt.legend()


# ROBOT SIMULATION
mobile_robot = DDMR()  # robot initialization
init_state = np.array([obs_path_nocbf[0,0], obs_path_nocbf[0,1], 1.589])  # initial state
mobile_robot.set_state(state = init_state)  # set the initial state
state_rec = []  # state record list
state_rec.append(init_state)  # record the initial state
mode_rec = []  # mode record list
mode_rec.append('grip')  # record the initial mode
for i in range(1,len(tVec)):
    state, mode = mobile_robot.dynamics_step(dt = 0.01, u = np.array([vx_ref[i], omega_ref[i]]))  # perform a dynamics step
    state_rec.append(state)  # record the state
    mode_rec.append(mode)  # record the mode

state_rec = np.array(state_rec)  # convert the list to a numpy array
mode_rec = np.array(mode_rec)  # convert the list to a numpy array

plt.figure()
plt.plot(state_rec[:,0],state_rec[:,1],'b-',label = 'robot path (no cbf)')
plt.plot(obs_path_nocbf[:,0],obs_path_nocbf[:,1],'k--',label = 'DMP obs. + nocbf')
plt.legend()

plt.figure()
plt.plot(tVec,state_rec[:,2],'b-',label = 'orient (no cbf)')
plt.legend()

plt.figure()
plt.plot(tVec,mode_rec,'b-',label = 'mode (no cbf)')
plt.legend()

plt.show()
print(">> End of the script")

