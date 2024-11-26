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
    
    def compute_u_safe_dmp_traj(self, dmp_traj, alpha, mu_s, g, exp):
        
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
        f3 = (K1*(xg-x) - D1*dx) / self.tau
        f4 = (K2*(yg-y) - D2*dy) / self.tau 

        # Input mapping
        G = np.array([[0,0],[0,0],[1,0],[0,1]]) 

        # Forcing term of the system
        f = self.compute_forcing_term(dmp_traj) / self.tau  # forcing terms
        forc_term_1 = f[0]  # forcing term for x
        forc_term_2 = f[1]  # forcing term for y

        # Inputs
        u = np.array([[K1*forc_term_1-K1*(xg-x0)*dmp_traj.cs.s],[K2*forc_term_2-K2*(yg-y0)*dmp_traj.cs.s]])  # input vector

        # CBF 
        h = mu_s*g - ((x*dy-y*dx)**2)/((x**2+y**2)**(3./2.))  # constraint function
        # Gradient of h components
        dh1 = (3*x*((x*dy-y*dx)**2)-2*dy*(x*dy-y*dx)*((x**2+y**2)**3))/((x**2+y**2)**(5./2.))  # derivative of h with respect to x
        dh2 = (3*y*((x*dy-y*dx)**2)-2*dx*(x*dy-y*dx)*((x**2+y**2)**3))/((x**2+y**2)**(5./2.))  # derivative of h with respect to y
        dh3 = (2*y*(x*dy-y*dx))/((x**2+y**2)**(3./2.))  # derivative of h with respect to dx 
        dh4 = (-2*x*(x*dy-y*dx))/((x**2+y**2)**(3./2.))  # derivative of h with respect to dy
        # Lie derivatives of h
        Lfh = dh1*f1 + dh2*f2 + dh3*f3 + dh4*f4  # Lie derivative of h
        # Lie derivatives of g
        Lgh = np.matmul(np.array([dh1, dh2, dh3, dh4]), G)
        Lghu = np.matmul(Lgh, u)  # Lie derivative of h with respect to u

        dot_h = Lfh + Lghu  # time derivative of h
        Psi = dot_h + alpha*(h**exp)
        K_psi = - np.nan_to_num(Lgh.T / (Lgh @ Lgh.T))

        # u_safe
        if Psi >= 0:
            u_safe = np.array([0, 0])
        elif Psi < 0:
            u_safe = K_psi.T * Psi
        return u_safe, Psi


        # K_v = dmp_traj.K
        # K_w = dmp_traj.K
        # D_v = dmp_traj.D
        # D_w = dmp_traj.D
        # omega = dmp_traj.dx[1]
        # v_r = dmp_traj.dx[0]
        # v_t = omega * dmp_traj.x[0] #CORRECT
        # rho = dmp_traj.x[0]

        # f_1 = v_r / self.tau 
        # f2 = omega / self.tau
        # f_3 = (K_v * (dmp_traj.x_goal[0] - dmp_traj.x[0]) - D_v * v_r - K_v * (dmp_traj.x_goal[0] - dmp_traj.x_0[0]) * self.s)/self.tau
        # f_4 = (K_w * (dmp_traj.x_goal[1] - dmp_traj.x[1]) - D_w * omega - K_w * (dmp_traj.x_goal[1] - dmp_traj.x_0[1]) * dmp_traj.cs.s)/self.tau

        # f = self.compute_forcing_term(dmp_traj) / self.tau  # forcing terms
        # f_v = f[0]  # forcing term for v
        # f_omega = f[1]  # forcing term for omega

        psi = -omega**2*f_1-2*rho*omega*(f_4 + K_w * f_omega) + alpha * (mu_s*g - v_t*omega)**exp

        # u_safe
        if psi >= 0:
            u_safe = np.array([0, 0])  # this includes the case when v_x = 0 or omega = 0
            status = True
            return u_safe, psi
        else:
            status = False
            u_safe = np.array([0, 1/(2*rho*omega)])*psi
            return u_safe, psi
        
    # def compute_u_safe_dmp_traj(self, dmp_traj, alpha, mu_s, g, exp):
    #     #K_v = dmp_traj.K
    #     K_w = dmp_traj.K
    #     #D_v = dmp_traj.D
    #     D_w = dmp_traj.D
    #     omega = dmp_traj.dx[1]
    #     v_r = dmp_traj.dx[0]
    #     v_t = omega * dmp_traj.x[0] #CORRECT
    #     rho = dmp_traj.x[0]

    #     f_1 = v_r / self.tau 
    #     # f2 = omega / self.tau
    #     # f_3 = (K_v * (dmp_traj.x_goal[0] - dmp_traj.x[0]) - D_v * v_r - K_v * (dmp_traj.x_goal[0] - dmp_traj.x_0[0]) * self.s)/self.tau
    #     f_4 = (K_w * (dmp_traj.x_goal[1] - dmp_traj.x[1]) - D_w * omega - K_w * (dmp_traj.x_goal[1] - dmp_traj.x_0[1]) * dmp_traj.cs.s)/self.tau

    #     f = self.compute_forcing_term(dmp_traj) / self.tau  # forcing terms
    #     f_v = f[0]  # forcing term for v
    #     f_omega = f[1]  # forcing term for omega

    #     psi = -omega**2*f_1-2*rho*omega*(f_4 + K_w * f_omega) + alpha * (mu_s*g - v_t*omega)**exp

    #     # u_safe
    #     if psi >= 0:
    #         u_safe = np.array([0, 0])  # this includes the case when v_x = 0 or omega = 0
    #         status = True
    #         return u_safe, psi
    #     else:
    #         status = False
    #         u_safe = np.array([0, 1/(2*rho*omega)])*psi
    #         return u_safe, psi


        

        
