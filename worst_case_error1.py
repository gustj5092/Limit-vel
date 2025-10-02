import numpy as np

# Module 3: Worst-case Estimation Error Computation
# File: worst_case_error.py

def compute_e_star(P: np.ndarray, a: np.ndarray, beta: float) -> np.ndarray:

    D = a.T @ P @ a
    e_star = -np.sqrt(beta) / np.sqrt(D) * (P @ a)
    return e_star

def compute_e_dot_star(P: np.ndarray, P_dot: np.ndarray, a: np.ndarray, a_dot: np.ndarray, beta: float) -> np.ndarray:

    D = a.T @ P @ a
    sqrt_beta = np.sqrt(beta)
    sqrt_D = np.sqrt(D)

    # First term
    term1 = -sqrt_beta / sqrt_D * (P_dot @ a)

    # Second term numerator (scalar)
    scalar_term = a.T @ P_dot @ a
    term2 = (sqrt_beta / (2 * D ** (3/2))) * (P @ a) * scalar_term

    # Final result
    e_dot_star = term1 + term2
    return e_dot_star