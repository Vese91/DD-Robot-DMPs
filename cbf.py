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
    
    def compute_u_safe_dmp_traj(self, dmp_traj, alpha, mu_s, exp, k):
        K_v = dmp_traj.K
        K_w = dmp_traj.K
        D_v = dmp_traj.D
        D_w = dmp_traj.D
        omega = dmp_traj.dx[1]
        v_r = dmp_traj.dx[0]
        v_t = omega * dmp_traj.x[0] #CORRECT
        rho = dmp_traj.x[0]
        self.update_s(dmp_traj) #as in step function

        # f_1 = v_r / self.tau = omega * rho / self.tau
        f_3 = K_v * (dmp_traj.x_goal[0] - dmp_traj.x[0]) - D_v * v_r - K_v * (dmp_traj.x_goal[0] - dmp_traj.x_0[0]) * self.s
        f_4 = K_w * (dmp_traj.x_goal[1] - dmp_traj.x[1]) - D_w * omega - K_w * (dmp_traj.x_goal[1] - dmp_traj.x_0[1]) * self.s

        f = self.compute_forcing_term(dmp_traj)
        f_v = f[0]
        f_omega = f[1]

        # u_safe parameters
        g = 9.81  # gravity
        h1 = np.sqrt(v_t**2 * omega**2 + k)  # h(x) = mu_s g - h1(x)
        psi = -(v_t*omega**2)/h1 * (f_3 + K_v*f_v) - (v_t**2*omega)/h1 * (f_4 + K_w*f_omega) + alpha * (mu_s*g - h1)**exp 

        # u_safe
        if psi >= 0:
            u_safe = np.array([0, 0])  # this includes the case when v_x = 0 or omega = 0
            return u_safe
        else:
            u_safe = h1**2/(v_t**2 * omega**4 + v_t**4 * omega**2) * psi * np.array([(v_t*omega**2)/h1, (v_t**2*omega)/h1])
            return u_safe


        #if (omega * v_x) > 0:
        #    psi_plus = - omega * (f_3 + K_v * f_v) - v_x * (f_4 + K_w * f_omega) + alpha * ((mu_s * g - v_x * omega))**exp
        #    if psi_plus >= 0:
        #        return np.array([0, 0])
        #    else:
        #        u_safe = 1 / (v_x**2 + omega**2) * psi_plus * np.array([omega, v_x])
        #        return u_safe
        #elif (omega * v_x) < 0:
        #    psi_minus = omega * (f_3 + K_v * f_v) + v_x * (f_4 + K_w * f_omega) + alpha * ((mu_s * g + v_x * omega))**exp
        #    if psi_minus >= 0:
        #        return np.array([0, 0])
        #    else:
        #        u_safe = -1 / (v_x**2 + omega**2) * psi_minus * np.array([omega, v_x])
        #        return u_safe
        #else:
        #    return np.array([0, 0])

        
