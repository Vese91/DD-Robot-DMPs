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

def create_train_vel(t, vx_max, omega_max, perc = 1):
    '''
    Create the training velocity and angular velocity for the robot.
    '''
    # Forward training velocity
    vx_1 = perc*vx_max*(2/(1+np.exp(-10*t[0:int(n/2)]))-1)  # training velocity
    vx_2 = np.flip(perc*vx_max*(2/(1+np.exp(-10*t[0:int(n/2)+1]))-1))  # flip the array
    vx = np.concatenate((vx_1,vx_2))  # concatenate the two arrays

    # Angular training velocity
    #omega = perc*omega_max*np.sin(t)  # training angular velocity
    omega = perc*omega_max*np.exp(-(t-0.5*t[-1])**2)

    return vx, omega

def compute_D1(n, dt):
    '''
    Compute the matrices used to estimate the first derivative.
      n float  : dimensionality
      dt float : timestep
    '''

    d1_p = np.ones([n - 1])
    d1_p[0] = 4.
    d1_m = - np.ones([n-1])
    d1_m[-1] = - 4.
    D1 = sparse.diags((d1_p, d1_m), [1, -1]).toarray()
    D1[0,0] = - 3.
    D1[0, 2] = -1.
    D1[-1, -3] = 1.
    D1[-1,-1] = 3.
    D1 /= 2 * dt
    return D1

# Time vector
t0 = 0.0  # initial time
tf = 6  # final time
n = 1000  # discretization points
tVec = np.linspace(t0,tf,n+1)  # time vector (n+1 so that dt = tf/n is indeed tVec[1]-tVec[0])
dt = tf/n  # time step

# Training velocity for robot
vx_max = 2.20  # maximum forward velocity [m/s]
omega_max = 2.84  # maximum angular velocity [rad/s]
percentage = 0.80  # percentage of the maximum velocity
vx_train, omega_train = create_train_vel(tVec, vx_max ,omega_max, perc = percentage)  # training velocity

# TRAINING DATA
robot = DDMR()  # initialize the robot
x0 = 0.0  # initial x position
y0 = 1.0  # initial y position
theta0 = 0.0  # initial orientation
robot.set_state([x0, y0, theta0])  # set the robot state

# reference trajectory
ref_state = []  # reference state list (x, y, theta)
ref_state.append([x0, y0, theta0])  # append the initial state
for i in range(1,len(tVec)):
    robot.state, _ = robot.dynamics_step(dt, [vx_train[i], omega_train[i]])
    ref_state.append(robot.state)

ref_state = np.array(ref_state)  # reference state array

# DMPs TRAINING
training_path = ref_state[:,0:2]  # training path
n_bfs = 100  # number of basis functions
dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 70, dt = 0.01, T = tf,
                              alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs (K=115)
dmp_traj.imitate_path(x_des = training_path)  # train the DMPs

# DMPs EXECUTION (NO CBF)
dmp_traj.x_0 = np.array([1.0, 1.0])  # new start in cartesian coordinates
dmp_traj.x_goal = np.array([2.0, 0.5])  # new goal in cartesian coordinates
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

# Reference velocity and angular velocity
vx_nocbf = np.sqrt(vel_nocbf[:,0]**2+vel_nocbf[:,1]**2)
theta = np.arctan2(np.gradient(path_nocbf[:,1]), np.gradient(path_nocbf[:,0]))
theta = np.unwrap(theta)
D1 = compute_D1(len(theta), dt)
omega_nocbf = D1 @ theta  

# DMPs EXECUTION (WITH CBF)
mu_s = 0.52  # static friction coefficient
mu_d = 0.42  # dynamic friction coefficient (80% of static friction)
g = 9.81 # gravity acceleration [m/s^2]
alpha = 10 # extended class-K function parameter (straight line, default 50)
exp = 1 # exponent of the extended class-K function, it must be an odd number (leave it as 1)
K_appr = 0.0001  # approximation of the constraint (default 0.0001)

cbf = CBF()  # CBF initialization
dmp_traj.reset_state()  # reset the state of the DMPs
x_list = np.array(dmp_traj.x) # x, y
x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

goal_tol = 0.01 # goal tolerance
while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
    obs_force = np.array([0,0])  # obstacle force (no obstacles)
    external_force, psi = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha, mu_s, g, exp, obs_force, omega_max, K_appr)  # compute the external force
    x, x_dot, x_ddot = dmp_traj.step(external_force = external_force)  # execute the DMPs
    x_list = np.vstack((x_list, x))
    x_dot_list = np.vstack((x_dot_list, x_dot))
    x_ddot_list = np.vstack((x_ddot_list, x_ddot))

path_cbf = copy.deepcopy(x_list)
vel_cbf = copy.deepcopy(x_dot_list)

# Reference velocity and angular velocity
vx_cbf = np.sqrt(vel_cbf[:,0]**2+vel_cbf[:,1]**2)
theta = np.arctan2(np.gradient(path_cbf[:,1]), np.gradient(path_cbf[:,0]))
theta = np.unwrap(theta)
D1 = compute_D1(len(theta), dt)
omega_cbf = D1 @ theta  

# plt.figure(1)
# plt.plot(tVec,vx_train,'b', label = r'$v_x$')
# plt.plot(tVec,vx_max*np.ones(n+1),'r--', label = r'$v_{x,max}$')
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [m/s]')
# plt.legend()

# plt.figure(2)
# plt.plot(tVec,omega_train,'b', label = r'$\omega$')
# plt.plot(tVec,omega_max*np.ones(n+1),'r--', label = r'$\omega_{max}$')
# plt.xlabel('Time [s]')
# plt.ylabel('Angular velocity [rad/s]')
# plt.legend()

# plt.figure(3)
# plt.plot(vx_train*omega_train,'b', label = r'$F_{cf}$')
# plt.xlabel('Time [s]')
# plt.ylabel('Centrifugal force [N]')
# plt.legend()


plt.figure(5)
plt.plot(path_nocbf[:,0],path_nocbf[:,1],'r-', label = 'ref. trajectory') 
plt.plot(path_cbf[:,0],path_cbf[:,1],'b-', label = 'CBF trajectory')
plt.plot(path_nocbf[0,0],path_nocbf[0,1],'go', label = 'initial pos.')
plt.plot(path_nocbf[-1,0],path_nocbf[-1,1],'ro', label = 'goal pos.')
plt.plot(path_cbf[0,0],path_cbf[0,1],'go', label = 'initial pos.')
plt.plot(path_cbf[-1,0],path_cbf[-1,1],'ro', label = 'goal pos.')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()


plt.figure(6)
plt.subplot(2,1,1)
plt.plot(vx_nocbf,'r-',label = 'ref. velocity')
plt.plot(vx_cbf,'b-',label = 'CBF velocity')

plt.subplot(2,1,2)
plt.plot(omega_nocbf,'r-',label = 'ref. angular velocity')
plt.plot(omega_cbf,'b-',label = 'CBF angular velocity')


plt.figure(7)
plt.plot(vx_nocbf*omega_nocbf,'r-',label = 'ref. centrifugal force')
plt.plot(vx_cbf*omega_cbf,'b-',label = 'CBF centrifugal force')

plt.show()


print(">>  End of the script")