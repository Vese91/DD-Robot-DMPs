import numpy as np

class DDMR(object):
    '''
    Differential-Drive Mobile Robot (DDMR) class.
    '''
    def __init__(self, state = np.zeros(6), dt = 0.01, m = 0.0, I = 0.0):
        '''
        Constructor for the DDMR class.
        '''
        self.state = state  # robot state
        self.dt = dt  # time step
        self.m = m  # mass
        self.I = I  # inertia

    def inverse_dynamics(self, vx, omega):
        '''
        Calculate the control input using the inverse dynamics.

        Inputs:
            self: class object (state, time-step, mass, inertia)
            vx: forward velocity (reference)
            omega: angular velocity (reference)

        Outputs:
            F: driving force
            T: driving torque
        '''
        F = self.m*vx/self.dt  # driving force
        T = self.I*omega/self.dt  # driving torque
        return F, T

    def dynamics_step(self, u = np.zeros(2)):
        '''
        Calculate the next state of the robot using the dynamics.

        Inputs:
            self: class object (state, time-step, mass, inertia)
            u: control input (driving force, driving torque)
        
        Outputs:
            state: next state of the robot
            mode: current mode of the robot (grip or slip)
        '''
        # Unpack the state
        x = self.state[0]  # x position
        y = self.state[1]  # y position
        theta = self.state[2]  # orientation
        vx = self.state[3]  # forward velocity
        vy = self.state[4]  # lateral velocity
        omega = self.state[5]  # angular velocity
        state_0 = np.array([x, y, theta, vx, vy, omega])  # current state

        # Unpack the control input
        F = u[0]  # driving force
        T = u[1]  # driving torque
        
        # Friction coefficients
        crr = 0.08  # rolling resistance coefficient
        srr = 0.05  # steering resistance coefficient
        mu_s = 0.70  # static friction coefficient
        mu_d = 0.56  # dynamic friction coefficient (80% of static friction)
        g = 9.81  # gravity acceleration [m/s^2]
        vy_critical = 0.001 # critical lateral velocity to switch back to grip state

        # Hybrid two-state system
        mode = 'grip'
        if mode.lower() == 'grip':
            # Grip state
            f = np.array([vx*np.cos(theta), vx*np.sin(theta), omega, 0, 0, 0])  # drift term
            B = np.array([[0,0], [0,0], [0,0], [1/self.m,0], [0,0], [0,1/self.I]])  # control matrix
            F_roll = np.sign(vx)*crr*self.m*g  # rolling resistance force
            T_roll = np.sign(omega)*srr*self.I  # steering resistance torque

            # Dynamic step
            u_effective = np.array([F-F_roll, T-T_roll])  # rolling resistance input
            self.state = state_0 + self.dt*(f + np.matmul(B, u_effective))  # state update
            self.state[4] = 0 # lateral velocity reset (it avoids unwanted lateral drift)

            # Transition condition
            F_cf = self.m*self.state[3]*self.state[5]  # centrifugal force
            F_cp = np.sign(F_cf)*mu_s*self.m*g  # centrifugal force limit (centripetal force)
            if np.abs(F_cf) > np.abs(F_cp):
                mode = 'slip'  # switch to slip state
        elif mode.lower() == 'slip':
            # Slip state
            f = np.array([vx*np.cos(theta)-vy*np.sin(theta), vx*np.sin(theta)+vy*np.cos(theta), omega, vy*omega, -vx*omega, 0])  # drift term
            B = np.array([[0,0,0],[0,0,0],[0,0,0],[1/self.m,0,0],[0,1/self.m,0],[0,0,1/self.I]])  # control matrix
            F_roll = np.sign(vx)*crr*self.m*g  # rolling resistance force
            T_roll = np.sign(omega)*srr*self.I  # steering resistance torque
            F_lat = -np.sign(vy)*mu_d*self.m*g  # lateral friction force

            # Dynamic step
            u_effective = np.array([F-F_roll, F_lat, T-T_roll])  # rolling resistance input
            self.state = state_0 + self.dt*(f + np.matmul(B, u_effective))

            # Transition condition
            if np.abs(self.state[4]) < vy_critical:
                mode = 'grip'  # switch to grip state
        
        return self.state, mode
        
    
    
    
        