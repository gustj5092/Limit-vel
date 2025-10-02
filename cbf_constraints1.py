import numpy as np
from scipy import sparse
import osqp
from scipy.stats import norm

def cbf_qp_osqp(h_val, lfh_val, lgh_val, u_des, v_max, R_cbf, gamma=1.0, osqp_opts=None):
    """
    OSQP를 사용하여 일반적인 단일 CBF 제약조건을 푸는 QP 솔버.
    제약: Lfh + Lgh*u + gamma*h >= 0  => -Lgh*u <= Lfh + gamma*h
    """
    u_des = float(np.asarray(u_des).reshape(()))
    
    # OSQP 표준 형식: 0.5 u^T P u + q^T u,  l <= A u <= u
    P = sparse.csc_matrix([[2.0 * R_cbf[0,0]]]) 
    q = np.array([-2.0 * R_cbf[0,0] * u_des])
    
    A = sparse.csc_matrix([[-lgh_val]])
    l = np.array([-np.inf])
    u_vec = np.array([lfh_val + gamma * h_val])

    if osqp_opts is None:
        osqp_opts = dict(eps_abs=1e-8, eps_rel=1e-8, max_iter=10000, polish=True, verbose=False)
        
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u_vec, **osqp_opts)
    res = prob.solve()

    status = (res.info.status or "").lower()
    if ("solved" in status) and (res.x is not None):
        return float(res.x[0])

    # 실패 시 안전한 fallback
    return float(np.clip(u_des, -np.inf, u_vec[0] / (-lgh_val) if lgh_val != 0 else np.inf))


def our_cbf_qp(x_hat, e_star, e_dot_star, u_des, v_max, R_cbf):
    """ OURS (TVRCBF) """
    v = x_hat[1] + e_star[1]
    v_dot = e_dot_star[1] 
    
    h = v_max - v
    lfh = -v_dot
    lgh = -1.0
    
    return cbf_qp_osqp(h, lfh, lgh, u_des, v_max, R_cbf)

def mrcbf_cbf_qp(x_hat, u_des, v_max, R_cbf, L_alpha_h, epsilon):
    """ MR-CBF """
    v = x_hat[1]
    h = v_max - v
    
    # 제약: Lfh + Lgh*u - L_alpha_h*eps >= -alpha*h
    # (Lfh=0, Lgh=-1, alpha=1.0) -> -u - L_alpha_h*eps >= -h
    # -> u <= h - L_alpha_h*eps
    u_upper_bound = h - L_alpha_h * epsilon
    
    return min(u_des, u_upper_bound)


def cccbf_cbf_qp(x_hat, u_des, v_max, R_cbf, Pk, delta=0.05):
    """ CC-CBF """
    v = x_hat[1]
    h = v_max - v
    
    # 제약: -u + h >= z*sqrt(Pk_vv)
    # -> u <= h - z*sqrt(Pk_vv)
    z = float(norm.ppf(1.0 - float(delta)))
    Pk_vv = Pk[1, 1]
    
    u_upper_bound = h - z * np.sqrt(max(Pk_vv, 0.0))
    
    return min(u_des, u_upper_bound)