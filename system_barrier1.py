import numpy as np

'''
def acc_dynamics(x, u):
    """Double integrator ACC dynamics."""
    dx1 = x[1]
    dx2 = u
    return np.array([dx1, dx2])
'''
def linear_barrier(x, a, b):
    """Linear barrier function h(x) = a^T x + b."""
    return np.dot(a, x) + b

def f_sample(x):
    """Drift dynamics f(x): no control part"""
    return np.array([x[1], 0.0])

def C_sample(x):
    """Measurement model: measure position only (x[0])"""
    return np.array([x[0], x[1]])

def acc_dynamics_with_noise(x, u, Q):
    """
    ACC dynamics with process noise:
    dx = [x2, u] + w(t), where w(t) ~ N(0, Q)
    """
    dx_nominal = np.array([x[1], u])
    w = np.random.multivariate_normal(mean=np.zeros(len(x)), cov=Q)
    return dx_nominal + w
