import numpy as np

class DDMR(object):
    '''
    Differential-Drive Mobile Robot (DDMR) class.
    '''
    def __init__(self, state = np.zeros(6), mode = 'grip', vy_in = 0.0, vx_max = 1.0, omega_max = 2.83, mu_s = 0.70, mu_d = 0.56, g = 9.81):
        '''
        Class constructor.
        '''
        self.state = state  # robot state (x,y,theta,vx,vy,w)
        self.mode = mode  # robot mode (grip, slip)
        self.vy_in = vy_in  # robot lateral velocity 
        self.vx_max = vx_max  # maximum forward velocity
        self.omega_max = omega_max  # maximum angular velocity
        self.mu_s = mu_s  # static friction coefficient
        self.mu_d = mu_d  # dynamic friction coefficient
        self.g = g  # gravity constant [m/s^2]

    def set_state(self, state):
        '''
        Set robot state
        '''
        self.state = state

    def get_ref_velocity(self, Tf = 1.0, path = np.zeros([10,2]), vel = np.zeros([10,2])):
        '''
        None
        '''
        # Time vector
        tVec = np.linspace(0,Tf,len(path))

        # Reference forward velocity
        vx_ref = np.sqrt(vel[:,0]**2+vel[:,1]**2)

        # Reference angular velocity
        theta = np.arctan2(np.gradient(path[:,1]), np.gradient(path[:,0]))
        theta = np.unwrap(theta)
        omega_ref = np.gradient(theta)/np.gradient(tVec)

        return vx_ref, omega_ref
    
    def forward_kinematics(self, u = np.zeros(2)):
        '''
        Function to calculate the forward kinematics of the DDMR.

        Inputs:
            self: object
            u: control input (vx, omega)
        
        Outputs:
            x_dot: x velocity
            y_dot: y velocity
            theta_dot: angular velocity
        '''
        # Unpack the state
        x = self.state[0]  # x position
        y = self.state[1]  # y position
        theta = self.state[2]  # orientation

        # Unpack the control input
        vx = u[0]  # forward velocity
        omega = u[1]  # angular velocity

        # Calculate the forward kinematics
        x_dot = vx * np.cos(theta)  # x velocity
        y_dot = vx * np.sin(theta)  # y velocity
        theta_dot = omega  # angular velocity

        return x_dot, y_dot, theta_dot
    
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
        K_appr = 0.0001  # approximation constant  # approximation constant
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

            # Check if the robot is slipping
            if vx_in > np.sqrt(self.mu_s**2*self.g**2-K_appr)/self.omega_max:
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