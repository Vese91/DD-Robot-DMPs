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

        psi = dmp_traj.gen_psi(self.s)
        f = dmp_traj.w @ psi[:, 0] / (np.sum(psi[:, 0])) * self.s
        f = np.nan_to_num(M @ f)
        return f
    
    def compute_u_safe_dmp_traj(self, dmp_traj, alpha, mu_s):
        K_v = dmp_traj.K
        K_w = dmp_traj.K
        D_v = dmp_traj.D
        D_w = dmp_traj.D
        v_x = dmp_traj.dx[0]
        omega = dmp_traj.dx[1]
        self.update_s(dmp_traj) #as in step function

        f_3 = K_v * (dmp_traj.x_goal[0] - dmp_traj.x[0]) - D_v * v_x - K_v * (dmp_traj.x_goal[0] - dmp_traj.x_0[0]) * self.s
        f_4 = K_w * (dmp_traj.x_goal[1] - dmp_traj.x[1]) - D_w * omega - K_w * (dmp_traj.x_goal[1] - dmp_traj.x_0[1]) * self.s

        f = self.compute_forcing_term(dmp_traj)
        f_v = f[0]
        f_omega = f[1]

        g = 9.81

        if omega > 0:
            psi_plus = - omega * (f_3 + K_v * f_v) - v_x * (f_4 + K_w * f_omega) + alpha * (mu_s * g - v_x * omega)
            if psi_plus >= 0:
                return np.array([0, 0])
            else:
                u_safe = 1 / (v_x**2 + omega**2) * psi_plus * np.array([omega, v_x])
                return u_safe
        elif omega == 0:
            return np.array([0, 0])
        elif omega < 0:
            psi_minus = omega * (f_3 + K_v * f_v) + v_x * (f_4 + K_w * f_omega) + alpha * (mu_s * g + v_x * omega)
            if psi_minus >= 0:
                return np.array([0, 0])
            else:
                u_safe = -1 / (v_x**2 + omega**2) * psi_minus * np.array([omega, v_x])
                return u_safe
