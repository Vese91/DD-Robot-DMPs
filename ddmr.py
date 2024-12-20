import numpy as np

class DDMR(object):
    '''
    Differential-Drive Mobile Robot (DDMR) class.
    '''
    def __init__(self, state = np.zeros(6), mode = 'grip', vy_in = 0.0, mu_s = 0.70, mu_d = 0.56, g = 9.81):
        '''
        Class constructor.
        '''
        self.state = state  # robot state (x,y,theta,vx,vy,w)
        self.mode = mode  # robot mode (grip, slip)
        self.vy_in = vy_in  # robot lateral velocity 
        self.mu_s = mu_s  # static friction coefficient
        self.mu_d = mu_d  # dynamic friction coefficient
        self.g = g  # gravity constant [m/s^2]

    def set_state(self, state):
        '''
        Set robot state
        '''
        self.state = state

    def get_ddmr_refinputs(tVec, traj, vel):
        '''
        Function to calculate the reference velocity for the DDMR.
        
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
    
    def dynamics_step(self, dt = 0.01, u = np.zeros(2)):
        '''
        Function to perform a dynamics step.

        Inputs:
            self: object
            dt: time step
            u: control input (vx, omega)
        
        Outputs:
            state: updated state
            mode: updated mode
        '''
        vy_critical = 0.001  # critical slip velocity
        if self.mode == 'grip':
            # Unpack the state
            x = self.state[0]  # x position
            y = self.state[1]  # y position
            theta = self.state[2]  # orientation

            # Unpack the control input
            vx_in = u[0]  # forward velocity
            omega_in = u[1]  # angular velocity

            # Update the state
            state_0 = np.array([x,y,theta]) # current state
            self.state = state_0 + dt*np.array([vx_in * np.cos(theta), vx_in * np.sin(theta), omega_in])  # explicit Euler integration
            if abs(vx_in*omega_in) > self.mu_s*self.g:
                self.mode = 'slip'

            return self.state, self.mode
        
        elif self.mode == 'slip':
            # Unpack the state
            x = self.state[0]  # x position
            y = self.state[1]  # y position
            theta = self.state[2]  # orientation

            # Input velocities
            vx_in = u[0]  # forward velocity
            omega_in = u[1]  # angular velocity
            self.vy_in = self.vy_in + dt * (-vx_in * omega_in - np.sign(self.vy_in)*self.mu_d*self.g)  # slip velocity
            
            # Update the state
            state_0 = np.array([x,y,theta])  # current state
            self.state = state_0 + dt*np.array([vx_in*np.cos(theta)-self.vy_in*np.sin(theta), vx_in*np.sin(theta)+self.vy_in*np.cos(theta), omega_in])  # explicit Euler integration
            if np.abs(self.vy_in) < vy_critical:
                self.mode = 'grip'

            return self.state, self.mode

        
        