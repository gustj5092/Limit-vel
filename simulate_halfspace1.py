import numpy as np
from scipy.integrate import solve_ivp
from worst_case_error1 import compute_e_star, compute_e_dot_star
from cbf_constraints1 import our_cbf_qp, mrcbf_cbf_qp, cccbf_cbf_qp
from state_estimator1 import kalman_filter_update
from system_barrier1 import f_sample, C_sample

def system_dynamics(x, u, w=None):
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    return A @ x + B.flatten() * u

def simulate(mode, x0, N, dt, Q, R, v_max, R_cbf, beta, w_list=None, v_list=None, lipschitz_constants=None, delta_cc=None):
    n = len(x0)
    x_real = np.zeros((N, n))
    x_hat = np.zeros((N, n))
    u_hist = np.zeros(N)
    h_real_hist = np.zeros(N)
    h_hat_hist = np.zeros(N)
    
    P_log = []
    x = x0.copy()
    xh = x0.copy()
    Pk = np.eye(n) * 0.1

    if mode == "mrcbf" and lipschitz_constants is None:
        raise ValueError("Lipschitz constants must be provided for MR-CBF mode.")
    if mode == "cccbf" and delta_cc is None:
        raise ValueError("delta_cc must be provided for CC-CBF mode.")

    for t in range(N):
        w = w_list[t] if w_list is not None else np.random.multivariate_normal(np.zeros(n), Q)
        v = v_list[t] if v_list is not None else np.random.multivariate_normal(np.zeros(n), R)
        
        # 칼만 필터 공분산 동역학
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        Cx = np.array([[1.0, 0.0], [0.0, 1.0]])
        K = Pk @ Cx.T @ np.linalg.inv(R)
        P_dot = A @ Pk + Pk @ A.T + Q - Pk @ Cx.T @ np.linalg.inv(R) @ Cx @ Pk


        u_des = 3.0 - xh[1] 
        if mode == "our":
            a = np.array([0.0, -1.0])
            a_dot = np.zeros_like(a)
            e_star = compute_e_star(Pk, a, beta)
            e_dot_star = compute_e_dot_star(Pk, P_dot, a, a_dot, beta)
            u = our_cbf_qp(xh, e_star, e_dot_star, u_des, v_max, R_cbf)
        
        elif mode == "mrcbf":
            _, L_alpha_h, _ = lipschitz_constants
            epsilon = np.sqrt(beta) * np.sqrt(np.max(np.linalg.eigvalsh(Pk))) # Worst-case error
            u = mrcbf_cbf_qp(xh, u_des, v_max, R_cbf, L_alpha_h, epsilon)

        elif mode == "cccbf":
            u = cccbf_cbf_qp(xh, u_des, v_max, R_cbf, Pk, delta=delta_cc)
            
        else: # "nom" 
             a = np.array([0.0, -1.0])
             h = v_max - (a @ xh)
             lfh = a @ f_sample(xh)
             lgh = a @ np.array([0., 1.])
             u = cbf_qp_osqp(h, lfh, lgh, u_des, v_max, R_cbf)


        # 시스템 및 추정기 업데이트
        x += dt * (system_dynamics(x, u)) + np.sqrt(dt) * w
        y = C_sample(x) + v
        
        xh += dt * (system_dynamics(xh, u) + K @ (y - C_sample(xh)))
        Pk += dt * P_dot

        e_v = x[1] - xh[1]
        Pk_vv = Pk[1, 1]
        if (e_v**2 / Pk_vv) > beta:
            e_v_proj = np.sqrt(beta / (e_v**2 / Pk_vv)) * e_v
            x[1] = xh[1] + e_v_proj

        # 로깅
        x_real[t] = x
        x_hat[t] = xh
        u_hist[t] = u
        h_hat_hist[t] = v_max - xh[1]
        h_real_hist[t] = v_max - x[1]
        P_log.append(Pk.copy())

    return {
        "x": x_real,
        "x_hat": x_hat,
        "u": u_hist,
        "h_hat": h_hat_hist,
        "h_real": h_real_hist,
        "P": P_log
    }