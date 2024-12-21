import numpy as np 
import copy 
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation
from dmp import dmp, obstacle_superquadric as obs
import bezier_interp as bz
from cbf import CBF
import math





# CONSTANTS
mu_s = 0.7  # static friction coefficient
g = 9.81  # gravity acceleration




def animate(x_list, obst_centers, obstacle_axis):  
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-')
    def update(frame):
        line.set_data(x_list[frame,0], x_list[frame,1])
        #also draw the line before
        if frame > 0:
            line.set_data(x_list[:frame,0], x_list[:frame,1])
        #also draw the obstacles as circles
        circle = plt.Circle(obst_centers[frame], obstacle_axis[0], color='g', fill=False)
        ax.add_artist(circle)
        #remove circle from previous iteration
        if frame > 0:
            circle = plt.Circle(obst_centers[frame-1], obstacle_axis[0], color='w', fill=False)
            ax.add_artist(circle)
        return line, ax

    ani = FuncAnimation(fig, update, frames=range(len(x_list)), blit=True)
    plt.show()




def get_ddmr_refinputs(tVec, traj, vel, dt = 0.01, mass = 1.0, inertia = 0.1, input_type = 'velocity'):
    '''
    Function to calculate the reference velocity for the DDMR
    
    Inputs:
        tVec: time vector (numpy array of shape (N,))
        traj: trajectory (numpy array of shape (N,2)) 
        vel: velocity (numpy array of shape (N,2))

    Outputs:
        vx_ref: reference forward velocity
        omega_ref: reference angular velocity
    '''
    # Calculate the tangential orientation and angular velocity
    orient = np.arctan2(np.gradient(traj[:, 1]), np.gradient(traj[:, 0]))  # orientation angle
    orient = np.unwrap(orient)  # unwrap the orientation angle (to avoid jumps)
    angvel = np.gradient(orient) / np.gradient(tVec)  # Calculate the angular velocity

    if input_type == 'velocity':
        vref = []  # reference velocity list
        for i in range(len(orient)):
            theta = orient[i]  # current orientation
            A = np.array([[np.cos(theta),np.sin(theta),0],[0,0,1]])  # inverse kinematics matrix
            b = np.array([vel[i,0], vel[i,1], angvel[i]])  # velocity vector in inertial frame
            vref_i = np.matmul(A,b)  # reference velocity
            vref.append(vref_i)  # reference velocity
        
        # Convert the list to a numpy array
        vref = np.array(vref)  # reference velocity (shape (N,2))
        vx_ref = vref[:,0]  # reference forward velocity
        omega_ref = vref[:,1]  # reference angular velocity

        return vx_ref, omega_ref
    
    elif input_type == 'force':
        vref = []  # reference velocity list
        for i in range(len(orient)):
            theta = orient[i]
            A = np.array([[np.cos(theta),np.sin(theta),0],[0,0,1]])  # inverse kinematics matrix
            b = np.array([vel[i,0], vel[i,1], angvel[i]])  # velocity vector in inertial frame
            vref_i = np.matmul(A,b)  # reference velocity
            vref.append(vref_i)  # reference velocity
        
        # Convert the list to a numpy array
        vref = np.array(vref)  # reference velocity (shape (N,2))
        vx_ref = vref[:,0]  # reference forward velocity
        omega_ref = vref[:,1]  # reference angular velocity

        # Calculate the linear acceleration and angular acceleration
        linear_acc = np.gradient(vx_ref) / dt
        angular_acc = np.gradient(omega_ref) / dt

        # Calculate the linear force and torque
        F_ref = mass * linear_acc
        T_ref = inertia * angular_acc

        return F_ref, T_ref
    
    else:
        raise ValueError('Invalid input type')

        return 0,0




def plot(x_list, x_dot_list, x_ddot_list, obstacle_centers = [], obstacle_axis = [], name = 'DMP', colour = 'b', lstyle = '-', mu_label = None):

    # Centrifugal force
    F_cf = ((x_list[:,0]*x_dot_list[:,1]-x_list[:,1]*x_dot_list[:,0])**2/((x_list[:,0]**2+x_list[:,1]**2)**(3/2)))

    # Reference inputs for the DDMR
    time_step = 0.01  # time step
    tVec = np.linspace(0, len(x_list)*time_step, len(x_list))  # time vector
    vx_ref, omega_ref = get_ddmr_refinputs(tVec, x_list, x_dot_list, dt = 0.01, mass = 1.0, inertia = 0.1, input_type = 'velocity')

    # Plot the result
    plt.subplot(2,2,1)
    plt.plot(x_list[:,0], x_list[:,1], color = colour, linestyle = lstyle, label = name, linewidth = 2.0)
    plt.plot(x_list[0,0], x_list[0,1], 'o')
    plt.plot(x_list[-1,0], x_list[-1,1],'x')
    # plot a circle for the obstacle
    # plot the enveloping of all the obstacles
    for i in range(len(obstacle_centers)):
        circle = plt.Circle(obstacle_centers[i], obstacle_axis[0], color = 'r', fill = False)
        plt.gca().add_artist(circle)
    plt.legend(fontsize = 12)
    plt.xlabel(r'$x$ [m]', fontsize = 15)
    plt.ylabel(r'$y$ [m]', fontsize = 15)
    plt.grid(True, linestyle = '--', linewidth = 0.5)

    plt.subplot(2,2,2)
    plt.plot(F_cf, color = colour, linestyle = lstyle, label = r'$F_{cf}$ ' + name, linewidth = 2.0)
    plt.axhline(y = mu_s * g, color = 'r', linestyle = '--', label = mu_label)
    plt.legend(fontsize = 12)
    plt.xlabel(r'$t$ [s]', fontsize = 15)
    plt.ylabel(r'$F$ [N]', fontsize = 15)
    plt.grid(True, linestyle = '--', linewidth = 0.5)

    plt.subplot(2,2,3)
    plt.plot(vx_ref, color = colour, linestyle = lstyle, label = r'$v_x$ ' + name, linewidth = 2.0)
    plt.legend(fontsize = 12)
    plt.xlabel(r'$t$ [s]', fontsize = 15)
    plt.ylabel(r'$v_x$ [m/s]', fontsize = 15)
    plt.grid(True, linestyle = '--', linewidth = 0.5)

    plt.subplot(2,2,4)
    plt.plot(omega_ref, color = colour, linestyle = lstyle, label = r'$\omega$ ' + name, linewidth = 2.0)
    plt.legend(fontsize = 12)
    plt.xlabel(r'$t$ [s]', fontsize = 15)
    plt.ylabel(r'$\omega$ [rad/s]', fontsize = 15)
    plt.grid(True, linestyle = '--', linewidth = 0.5)










def test(dmp_traj, start, goal, cbfs = True, obst = True):
    # INIT DMPs
    dmp_traj.x_0 = start  # new start in cartesian coordinates
    dmp_traj.x_goal = goal  # new goal in cartesian coordinates
    dmp_traj.reset_state()  # reset the state of the DMPs
    x_list = np.array(dmp_traj.x) # x, y
    x_dot_list = np.array(dmp_traj.dx)  # v_x, v_y
    x_ddot_list = np.array(dmp_traj.ddx)  # a_x, a_y

    obstacles = []
    obst_centers = []
    if obst:
        # OBSTACLES
        obstacle_center = np.array([0., -0.4])  # obstacle center
        obst_centers.append(obstacle_center)
        radius = 0.2
        obstacle_axis = np.ones(dmp_traj.n_dmps) * radius
        # superquadric parameters
        lmbda = 5.0
        beta = 2.5
        eta = 1.0
        obstacles.append(obs.Obstacle_Dynamic(center = obstacle_center, axis = obstacle_axis, 
                                lmbda = lmbda, beta=beta, eta=eta, coeffs = np.ones(dmp_traj.n_dmps)))
    
    if cbfs:
        # CBF PARAMETERS
        alpha = 75 # extended class-K function parameter (straight line) 
        exp = 1 # exponent of the extended class-K function, it must be an odd number (leave it as 1)
        cbf = CBF()
        
    # PLANNING LOOP
    goal_tol = 0.01 # goal tolerance
    iter = 0
    while not np.linalg.norm(dmp_traj.x - dmp_traj.x_goal) < goal_tol:
        external_force = np.zeros(dmp_traj.n_dmps)
        for obstacle in obstacles:
            curr_obst = copy.deepcopy(obstacle)
            freq_obst = 2
            amp_obst = 0.0
            vel_obst = np.array([0, amp_obst * 2*math.pi*2*np.cos(2*math.pi*freq_obst*iter*dmp_traj.cs.dt)])  # velocity of the curr_obst
            external_force += curr_obst.gen_external_force(dmp_traj.x, dmp_traj.dx - vel_obst)
            curr_obst.center += np.array([0, amp_obst * np.sin(2*math.pi*freq_obst*iter*dmp_traj.cs.dt)])  # move the curr_obst
            obst_centers.append(curr_obst.center)
        if cbfs:
            cbf_force, _ = cbf.compute_u_safe_dmp_traj(dmp_traj, alpha, mu_s, g, exp, external_force)
            external_force += cbf_force
        x, x_dot, x_ddot = dmp_traj.step(external_force=external_force)  # execute the DMPs        
        x_list = np.vstack((x_list, x))
        x_dot_list = np.vstack((x_dot_list, x_dot))
        x_ddot_list = np.vstack((x_ddot_list, x_ddot))

        iter += 1
    
    #generate dynamic plot of x_list with FuncAnimation
    # animate(x_list, obst_centers, obstacle_axis)

    return x_list, x_dot_list, x_ddot_list, obst_centers, obstacle_axis








def main():
    # REFERENCE TRAJECTORY (Cartesian coordinates) 
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

    # Colors list
    color_list = ['green','blue','darkorange']  # colors list (ref, cbf, no cbf)
    line_style = ['-','--']  # line style list

    # plt.plot(ref_path[:,0], ref_path[:,1],'b-',label='Reference trajectory')
    # plt.plot(ref_path[0,0], ref_path[0,1],'ko',label='Start')
    # plt.plot(ref_path[-1,0], ref_path[-1,1],'kx',label='Goal')
    # plt.title('Ref. training trajectory')
    # plt.legend()
    # plt.show()

    # DMPs TRAINING
    n_bfs = 100  # number of basis functions
    dmp_traj = dmp.DMPs_cartesian(n_dmps = 2, n_bfs = n_bfs, K = 115, dt = 0.01, T = t[-1],
                                alpha_s = 2.0, tol = 3.0 / 100, rescale = "rotodilatation", basis = "gaussian")  # set up the DMPs
    dmp_traj.imitate_path(x_des=ref_path)  # train the DMPs
    learnt_path, learnt_vel, learnt_acc, _ = dmp_traj.rollout(tau = 1)  # rollout the DMPs
    plot(learnt_path, learnt_vel, learnt_acc, name = 'training', colour = color_list[0], lstyle = line_style[0])

    #TEST
    x_list, x_dot_list, x_ddot_list, obst_centers, obst_axis = test(dmp_traj, start=np.array([-2, 1.5]), goal=np.array([3, -1.0]), cbfs=False, obst=True)  
    name = "DMP obst"
    plot(x_list, x_dot_list, x_ddot_list, obst_centers, obst_axis, name = name, colour = color_list[2], lstyle = line_style[0])
    x_list, x_dot_list, x_ddot_list, obst_centers, obst_axis = test(dmp_traj, start=np.array([-2, 1.5]), goal=np.array([3, -1.0]), cbfs=True, obst=True)
    #plot the result
    name = "DMP obst + cbf"
    plot(x_list, x_dot_list, x_ddot_list, obst_centers, obst_axis, name = name, colour = color_list[1], lstyle = line_style[0], mu_label = r'$\mu_s\,g$')
    plt.subplots_adjust(left = 0.086, right = 0.99, top = 0.99)  # trim settings
    plt.show()

if __name__ == "__main__":
    main()

print(">> End of the script")