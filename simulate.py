# simulate_halfspace.py
import numpy as np
from scipy.integrate import solve_ivp
from worst_case_error1 import compute_e_star, compute_e_dot_star
from cbf_constraints1 import nominal_cbf_qp, robust_cbf_qp, our_cbf_qp
from state_estimator1 import kalman_filter_update
from system_barrier1 import f_sample, C_sample

#def system_dynamics(x, u, w):
def system_dynamics(x, u, w=None):
    A = np.array([[0, 1], [0, 0]])
    B = np.array([[0], [1]])
    #return A @ x + B.flatten() * u + w
    return A @ x + B.flatten() * u 


def error_generator(P, beta, prev_e=None, decay=0.95, noise_scale=0.05, rng=np.random.default_rng()):
    """
    Generate a smooth estimation error e ∈ E_beta for velocity only (1D),
    where E_beta = { e | e[1]^2 / P[1,1] <= beta }

    Parameters:
    - P: covariance matrix (2x2)
    - beta: scalar chi² bound for velocity (df=1)
    - prev_e: previous full-state error (pos, vel)
    - decay, noise_scale: temporal smoothness and noise level

    Returns:
    - e: full-state error, but only velocity is clipped to tube
    """
    n = P.shape[0]
    if prev_e is None:
        prev_e = np.zeros(n)

    noise = rng.normal(scale=noise_scale, size=n)
    e_candidate = decay * prev_e + noise

    # Only velocity dimension (index 1) is used in h(x)
    v_error = e_candidate[1]
    P11 = P[1, 1]
    mahal_v = v_error**2 / P11

    if mahal_v > beta:
        e_candidate[1] *= np.sqrt(beta / mahal_v)

    return e_candidate


def simulate(mode, x0, N, dt, Q, R, v_max, R_cbf, beta, w_list=None, v_list=None):
    n = len(x0)
    x_real = np.zeros((N, n))
    x_hat = np.zeros((N, n))
    P = np.zeros((N, n, n))
    u_hist = np.zeros(N)
    h_real_hist = np.zeros(N)
    h_hat_hist = np.zeros(N)

    e_log = []
    edot_log = []
    P_log = []
    x = x0.copy()
    xh = x0.copy() #+ np.random.multivariate_normal(np.zeros(n), Q)
    Pk = np.eye(n) * 0.1

    e_prev = np.zeros(n)

    for t in range(N):
        time = t * dt

        if w_list is not None and v_list is not None:
            w = w_list[t]
            v = v_list[t]
        else:
            w = np.random.multivariate_normal(np.zeros(n), Q)
            v = np.random.multivariate_normal(np.zeros(n), R)


        A = np.array([[0.0, 1.0],
                    [0.0, 0.0]])  # df/dx
        Cx = np.array([[1.0, 0.0],
                        [0.0, 1.0]])  # dC/dx
        # Kalman gain
        K = Pk @ Cx.T @ np.linalg.inv(R)
        # P dot
        P_dot = A @ Pk + Pk @ A.T + Q- Pk @ Cx.T @ np.linalg.inv(R) @ Cx @ Pk
        if mode == "nom":
            u_des = 3.0 - xh[1]
            u = nominal_cbf_qp(xh, u_des, v_max, R_cbf)
            e_log.append(np.zeros(n))
            edot_log.append(np.zeros(n))

        elif mode == "rob":
            a = np.array([0.0, -1.0])
            e_star = compute_e_star(Pk, a, beta)
            u_des = 3.0 - (xh[1]+e_star[1])
            u = robust_cbf_qp(xh, e_star, u_des, v_max, R_cbf)
            e_log.append(e_star)
            edot_log.append(np.zeros(n))
            e = error_generator(Pk, beta, e_prev)
            x = xh + e
            e_prev = e


        elif mode == "our":
            a = np.array([0.0, -1.0])
            a_dot = np.zeros_like(a)
            e_star = compute_e_star(Pk, a, beta)

            # CKF 기반 P_dot 계산
            # x_hat_dot, P_dot = kalman_filter_update(xh, Pk, y, u_des, f_sample, C_sample, Q, R)

            e_dot_star = compute_e_dot_star(Pk, P_dot, a, a_dot, beta)
            u_des = 3.0 - (xh[1]+e_star[1])
            u = our_cbf_qp(xh, e_star, e_dot_star, u_des, v_max, R_cbf)
            e_log.append(e_star)
            edot_log.append(e_dot_star)
            e = error_generator(Pk, beta, e_prev)
            x = xh + e
            e_prev = e

        '''
        def ode_func(t_span, x_state):
            return system_dynamics(t_span, x_state, u, w)

        sol = solve_ivp(ode_func, [t, t + dt], x, t_eval=[t + dt])
        x = sol.y[:, -1]
        '''
        # state integration for true state
        def ode_true(t, x_state):
            return f_sample(x_state) + np.array([0.,1.])*u
        sol_true = solve_ivp(ode_true, [0, dt], x, method='RK45', atol=1e-8, rtol=1e-8)
        x = sol_true.y[:,-1] #+ w
        y = np.array([x[0], x[1]]) + v 

        # state integration for estimate xh
        def ode_est(t, xh_state):

            # use same measurement y and control u, assume Pk constant over dt
            K = Pk @ Cx.T @ np.linalg.inv(R)
            xh_dot = f_sample(xh_state) + np.array([0.0, 1.0]) * u + K @ (y - C_sample(xh_state))
            return xh_dot
        sol_est = solve_ivp(ode_est, [0, dt], xh, method='RK45', atol=1e-8, rtol=1e-8)
        xh = sol_est.y[:,-1]

        # CKF 기반 예측 상태 업데이트 (Euler integration)
        # x_hat_dot, P_dot = kalman_filter_update(xh, Pk, y, u, f_sample, C_sample, Q, R)
        # xh += dt * x_hat_dot
        Pk += dt * P_dot

        x_real[t] = x
        x_hat[t] = xh
        P[t] = Pk
        u_hist[t] = u
        h_hat_hist[t] = v_max - xh[1]  # 추정 기반 h
        h_real_hist[t] = v_max - x[1]  # 실제 상태 기반 h
        P_log.append(Pk.copy())
    return x_real, x_hat, u_hist, h_hat_hist, h_real_hist, e_log, edot_log, P_log