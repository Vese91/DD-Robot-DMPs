import numpy as np
from dmp import dmp

class CBF():
    def __init__(self, tau=1.) -> None:
        self.s = 1.0
        self.tau = tau
        pass

    def reset_s(self):
        self.s = 1.0

    def update_s(self, dmp_traj):
        #need to be done out of dmp, since step function already does it
        const = - dmp_traj.cs.alpha_s / self.tau
        self.s *= np.exp(const * dmp_traj.cs.dt)

    def compute_forcing_term(self, dmp_traj):
        if dmp_traj.rescale == 'rotodilatation':
            M = dmp.roto_dilatation(dmp_traj.learned_position, dmp_traj.x_goal - dmp_traj.x_0)
        elif dmp_traj.rescale == 'diagonal':
            M = np.diag((dmp_traj.x_goal - dmp_traj.x_0) / dmp_traj.learned_position)
        else:
            M = np.eye(dmp_traj.n_dmps)

        psi = dmp_traj.gen_psi(dmp_traj.cs.s)
        f = dmp_traj.w @ psi[:, 0] / (np.sum(psi[:, 0])) * dmp_traj.cs.s
        f = np.nan_to_num(M @ f)
        return f

    def compute_u_safe_dmp_traj(self, dmp_traj, v_max = 1.0, a_max = 5.0, delta_0 = 0.10, eta = 0.50, r_min = 0.25, gamma = 90.0, obs_center = np.array([0,0]),
                                alpha = 1.0, exp = 1.0, obs_force = np.array([0,0]), K_appr = 0.0001, type = 'velocity'):
        # Coefficients and variables
        K1 = dmp_traj.K  # spring constant 1
        K2 = dmp_traj.K  # spring constant 2
        D1 = dmp_traj.D  # damping constant 1
        D2 = dmp_traj.D  # damping constant 2
        x0 = dmp_traj.x_0[0]  # initial position (x-coordinate)
        y0 = dmp_traj.x_0[1]  # initial position (y-coordinate)
        xg = dmp_traj.x_goal[0]  # goal position (x-coordinate)
        yg = dmp_traj.x_goal[1]  # goal position (y-coordinate)
        x = dmp_traj.x[0]  # x position
        y = dmp_traj.x[1]  # y position
        dx = dmp_traj.dx[0]  # x velocity
        dy = dmp_traj.dx[1]  # y velocity

        # Drift term of the system
        f1 = dx / self.tau
        f2 = dy / self.tau
        f3 = (K1*(xg-x) - D1*dx + obs_force[0]) / self.tau
        f4 = (K2*(yg-y) - D2*dy + obs_force[1]) / self.tau

        # Input mapping
        G = np.array([[0,0],[0,0],[1,0],[0,1]])

        # Forcing term of the system
        f = self.compute_forcing_term(dmp_traj) / self.tau  # forcing terms
        forc_term_1 = f[0]  # forcing term for x
        forc_term_2 = f[1]  # forcing term for y  

        # Inputs  
        u = np.array([[K1*forc_term_1-K1*(xg-x0)*dmp_traj.cs.s],[K2*forc_term_2-K2*(yg-y0)*dmp_traj.cs.s]])  # input vector

        if type == 'velocity':   
           # Control Barrier Function (CBF)
           h = v_max - np.sqrt((dx**2+dy**2)+K_appr)  # constraint function 

           # Gradient of h components 
           dh1 = 0  # derivative of h with respect to x
           dh2 = 0  # derivative of h with respect to y
           dh3 = -dx/np.sqrt((dx**2+dy**2)+K_appr)  # derivative of h with respect to dx
           dh4 = -dy/np.sqrt((dx**2+dy**2)+K_appr)  # derivative of h with respect to dy

           # Lie derivatives of h
           Lfh = dh1*f1 + dh2*f2 + dh3*f3 + dh4*f4  # Lie derivative of h with respect to f
           Lgh = np.matmul(np.array([dh1, dh2, dh3, dh4]), G)  # Lie derivatives of g with respect to f
           Lghu = np.matmul(Lgh, u)  # Lgh * u

           # U_safe construction
           dot_h = Lfh + Lghu   # time derivative of h
           Psi = dot_h + alpha*(h**exp)  # safety function
           K_psi = - Lgh.T / (Lgh @ Lgh.T)  # gain matrix
           if Psi >= 0:
               u_safe = np.array([0, 0])
           elif Psi < 0:
               u_safe = K_psi.T * Psi
    
        elif type == 'force':
            # Control Barrier Function (CBF)
            h = a_max - np.sqrt((dy*x-dx*y)**4/(x**2+y**2)**3+K_appr)  # constraint function
            
            # Gradient of h components 
            dh1 = -((dy*x-dx*y)**3*(-dy*x**2+3*dx*x*y+2*dy*y**2))/((x**2+y**2)**4*(K_appr+(dy*x-dx*y)**4/(x**2+y**2)**3)**(1/2))
            dh2 = ((dy*x-dx*y)**3*(2*dx*x**2+3*dy*x*y-dx*y**2))/((x**2+y**2)**4*(K_appr+(dy*x-dx*y)**4/(x**2+y**2)**3)**(1/2))
            dh3 = (2*y*(dy*x-dx*y)**3)/((x**2+y**2)**3*(K_appr+(dy*x-dx*y)**4/(x**2+y**2)**3)**(1/2))
            dh4 = -(2*x*(dy*x-dx*y)**3)/((x**2+y**2)**3*(K_appr+(dy*x-dx*y)**4/(x**2+y**2)**3)**(1/2))

            # Lie derivatives of h
            Lfh = dh1*f1 + dh2*f2 + dh3*f3 + dh4*f4  # Lie derivative of h with respect to f
            Lgh = np.matmul(np.array([dh1, dh2, dh3, dh4]), G)  # Lie derivatives of g with respect to f
            Lghu = np.matmul(Lgh, u)  # Lgh * u

            # U_safe construction
            dot_h = Lfh + Lghu  # time derivative of h
            Psi = dot_h + alpha*(h**exp)  # safety function
            K_psi = - Lgh.T / (Lgh @ Lgh.T)  # gain matrix
            if Psi >= 0:
                u_safe = np.array([0, 0])
            elif Psi < 0:
                u_safe = K_psi.T * Psi

        elif type == 'obstacle':
            # Control Barrier Function (CBF)
            xO = obs_center[0]  # x-coordinate of the obstacle
            yO = obs_center[1]  # y-coordinate of the obstacle
            dxO = 0  # x velocity of the obstacle
            dyO = 0  # y velocity of the obstacle
            delta_0 = 0.05  # small constant for control barrier function
            eta = 0.25  # repulsive gain factor
            r_min = 0.25  # radius over which the repulsive potential field is active
            gamma = 100.0  # maximum acceleration for the robot

            # Repulsive potential field U(x,v)
            n_ro = (1/np.sqrt((xO-x)**2+(yO-y)**2))*np.array([[xO-x,yO-y]]).T  # unit vector pointing from the robot to the obstacle
            v_ro = np.array([dx-dxO,dy-dyO]) @ n_ro  # relative velocity between the robot and the obstacle
            rho_delta = np.sqrt((x-xO)**2+(y-yO)**2)-v_ro**2/(2*gamma)  # dynamic distance between the robot and the obstacle
            U_rep = eta*(1/rho_delta-1/r_min)  # repulsive potential field

            # Constraint function h(x)
            h = 1/(1+U_rep)-delta_0  # constraint function

            # Gradient of h(x) (components)
            dh1 = (eta*((x-xO)/((x-xO)**2+(y-yO)**2)**(1/2)+((2*x-2*xO)*(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO)**2)/(2*gamma*((x-xO)**2+(y-yO)**2)**2)-((dx-dxO)*(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO))/(gamma*((x-xO)**2+(y-yO)**2))))/((((x-xO)**2+(y-yO)**2)**(1/2)-(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO)**2/(2*gamma*((x-xO)**2+(y-yO)**2)))**2*(eta*(1/(((x-xO)**2+(y-yO)**2)**(1/2)-(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO)**2/(2*gamma*((x-xO)**2+(y-yO)**2)))-1/r_min)+1)**2)
            dh2 = (eta*((y-yO)/((x-xO)**2+(y-yO)**2)**(1/2)+((2*y-2*yO)*(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO)**2)/(2*gamma*((x-xO)**2+(y-yO)**2)**2)-((dy-dyO)*(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO))/(gamma*((x-xO)**2+(y-yO)**2))))/((((x-xO)**2+(y-yO)**2)**(1/2)-(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO)**2/(2*gamma*((x-xO)**2+(y-yO)**2)))**2*(eta*(1/(((x-xO)**2+(y-yO)**2)**(1/2)-(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO)**2/(2*gamma*((x-xO)**2+(y-yO)**2)))-1/r_min)+1)**2)
            dh3 = -(eta*(x-xO)*(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO))/(gamma*((x-xO)**2+(y-yO)**2)*(((x-xO)**2+(y-yO)**2)**(1/2)-(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO)**2/(2*gamma*((x-xO)**2+(y-yO)**2)))**2*(eta*(1/(((x-xO)**2+(y-yO)**2)**(1/2)-(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO)**2/(2*gamma*((x-xO)**2+(y-yO)**2)))-1/r_min)+1)**2)
            dh4 = -(eta*(y-yO)*(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO))/(gamma*((x-xO)**2+(y-yO)**2)*(((x-xO)**2+(y-yO)**2)**(1/2)-(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO)**2/(2*a_max*((x-xO)**2+(y-yO)**2)))**2*(eta*(1/(((x-xO)**2+(y-yO)**2)**(1/2)-(dx*x-dxO*x-dx*xO+dxO*xO+dy*y-dyO*y-dy*yO+dyO*yO)**2/(2*gamma*((x-xO)**2+(y-yO)**2)))-1/r_min)+1)**2)

            # Lie derivatives of h
            Lfh = dh1*f1 + dh2*f2 + dh3*f3 + dh4*f4  # Lie derivative of h with respect to f
            Lgh = np.matmul(np.array([dh1, dh2, dh3, dh4]), G)  # Lie derivatives of g with respect to f
            Lghu = np.matmul(Lgh, u)  # Lgh * u

            # U_safe construction
            dot_h = Lfh + Lghu  # time derivative of h
            Psi = dot_h + alpha*(h**exp)  # safety function
            K_psi = - Lgh.T / (Lgh @ Lgh.T)  # gain matrix
            if Psi >= 0:
                u_safe = np.array([0, 0])
            elif Psi < 0:
                u_safe = K_psi.T * Psi
        else:
            raise ValueError('Invalid type of CBF')

        return u_safe, Psi