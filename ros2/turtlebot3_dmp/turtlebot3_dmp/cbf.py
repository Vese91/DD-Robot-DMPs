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
    
    def compute_u_safe_dmp_traj(self, dmp_traj, alpha, mu_s, g, exp, obs_force = np.array([0,0])):
        
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

        # CBF 
        h = mu_s*g - ((x*dy-y*dx)**2)/((x**2+y**2)**(3./2.))  # constraint function
        # Gradient of h components
        dh1 = (3*x*((x*dy-y*dx)**2)-2*dy*(x*dy-y*dx)*((x**2+y**2)**3))/((x**2+y**2)**(5./2.))  # derivative of h with respect to x
        dh2 = (3*y*((x*dy-y*dx)**2)-2*dx*(x*dy-y*dx)*((x**2+y**2)**3))/((x**2+y**2)**(5./2.))  # derivative of h with respect to y
        dh3 = (2*y*(x*dy-y*dx))/((x**2+y**2)**(3./2.))  # derivative of h with respect to dx 
        dh4 = (-2*x*(x*dy-y*dx))/((x**2+y**2)**(3./2.))  # derivative of h with respect to dy
        # Lie derivatives of h
        Lfh = dh1*f1 + dh2*f2 + dh3*f3 + dh4*f4  # Lie derivative of h with respect to f
        Lgh = np.matmul(np.array([dh1, dh2, dh3, dh4]), G)  # Lie derivatives of g with respect to f 
        Lghu = np.matmul(Lgh, u)  # Lgh * u

        # U_safe construction
        dot_h = Lfh + Lghu  # time derivative of h
        Psi = dot_h + alpha*(h**exp)  # safety function
        K_psi = - Lgh.T / (Lgh @ Lgh.T)  # gain matrix
        # K_psi = - np.nan_to_num(Lgh.T / (Lgh @ Lgh.T))  # gain matrix
        if Psi >= 0:
            u_safe = np.array([0, 0])
        elif Psi < 0:
            u_safe = K_psi.T * Psi
        return u_safe, Psi