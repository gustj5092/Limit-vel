import numpy as np
from system_barrier1 import f_sample, C_sample
# Module 2: Kalman Filter (Continuous-Time)
# File: state_estimator.py

def kalman_filter_update(x_hat: np.ndarray, P: np.ndarray, y: np.ndarray, u: float,
                         f_func, C_func, Q: np.ndarray, R: np.ndarray) -> tuple:
    """
    Continuous Kalman Filter (CKF) update:
    - x_hat_dot = f(x_hat) + g(x_hat)·u + K(y - C(x_hat))
    - P_dot = A P + P Aᵀ + Q - P Cᵀ R⁻¹ C P

    Parameters:
    - x_hat: current state estimate (n,)
    - P: current covariance estimate (n x n)
    - y: current measurement (m,)
    - u: current control input (scalar)
    - f_func: function f(x) returning dynamics (n,)
    - C_func: function C(x) returning measurement (m,)
    - Q: process noise covariance (n x n)
    - R: measurement noise covariance (m x m)

    Returns:
    - x_hat_dot: derivative of x_hat (n,)
    - P_dot: derivative of P (n x n)
    """

    '''
    n = x_hat.shape[0]
    m = y.shape[0]

    # Numerical Jacobians
    eps = 1e-5
    A = np.zeros((n, n))
    Cx = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        A[:, i] = (f_func(x_hat + dx) - f_func(x_hat - dx)) / (2 * eps)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        Cx[:, i] = (C_func(x_hat + dx) - C_func(x_hat - dx)) / (2 * eps)
    '''

    A = np.array([[0.0, 1.0],
                  [0.0, 0.0]])  # df/dx
    Cx = np.array([[1.0, 0.0],
                   [0.0, 1.0]])  # dC/dx
    
    # Kalman gain
    K = P @ Cx.T @ np.linalg.inv(R)

    # x̂ dot
    x_hat_dot = f_func(x_hat) + np.array([0.0, 1.0]) * u + K @ (y - C_func(x_hat))

    # P dot
    P_dot = A @ P + P @ A.T + Q - P @ Cx.T @ np.linalg.inv(R) @ Cx @ P

    return x_hat_dot, P_dot