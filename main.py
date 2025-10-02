import numpy as np
import matplotlib.pyplot as plt
from simulate_halfspace1 import simulate
from scipy.stats import chi2, norm

# ----- 시뮬레이션 설정 -----
dt = 0.05
T = 15.0
N = int(T / dt)
x0 = np.array([0.0, 1.0])
Q = 0.0 * np.eye(2)
R = np.array([[0.05, 0], [0, 0.05]])
v_max = 3.0
R_cbf = np.array([[0.01]])

# 신뢰도 및 위험도 설정
confidence = 0.95
beta = chi2.ppf(confidence, df=1) # OURS (TVRCBF)
delta_cc = 1 - confidence        # CCCBF

# MR-CBF를 위한 립시츠 상수 
L_Lfh = 0.0
L_alpha_h = 1.0 
L_Lgh = 0.0
lipschitz_constants = (L_Lfh, L_alpha_h, L_Lgh)


# 노이즈 시퀀스 생성
np.random.seed(42)
w_list = [np.random.multivariate_normal(np.zeros(len(x0)), Q) for _ in range(N)]
v_list = [np.random.multivariate_normal(np.zeros(len(x0)), R) for _ in range(N)]

# ----- 모든 모드 실행 -----
modes = ["cccbf", "mrcbf", "our"]
results = {}

for mode in modes:
    print(f"--- Running: {mode.upper()} ---")
    results[mode] = simulate(
        mode, 
        x0, N, dt, Q, R, v_max, R_cbf, 
        beta=beta, 
        w_list=w_list, v_list=v_list,
        lipschitz_constants=lipschitz_constants,
        delta_cc=delta_cc
    )

# ----- 결과 플로팅 -----
time = np.linspace(0, T, N)
colors = {"cccbf": "forestgreen", "mrcbf": "dodgerblue", "our": "violet"}
labels = {"cccbf": "CCCBF", "mrcbf": "MRCBF", "our": "OURS (TVRCBF)"}


plt.figure(figsize=(10, 8))

# 속도 그래프
plt.subplot(3, 1, 1)
for mode in modes:
    plt.plot(time, results[mode]["x"][:, 1], label=labels[mode], color=colors[mode])
plt.axhline(v_max, color="red", linestyle=":", label="Velocity Limit")
plt.ylabel("Velocity")
plt.legend()
plt.title("Velocity over Time")

# 제어 입력 그래프
plt.subplot(3, 1, 2)
for mode in modes:
    plt.plot(time, results[mode]["u"], label=labels[mode], color=colors[mode])
plt.axhline(0.0, color="red", linestyle=":")
plt.ylabel("Control Input (u)")
plt.legend()
plt.title("Control Input over Time")

# CBF h(x) 그래프
plt.subplot(3, 1, 3)
for mode in modes:
    plt.plot(time, results[mode]["h_hat"], linestyle="-", label=f"h_hat ({labels[mode]})", color=colors[mode])
    plt.plot(time, results[mode]["h_real"], linestyle="--", label=f"h_real ({labels[mode]})", color=colors[mode]) # 실제 h(x)
plt.axhline(0, color="gray", linestyle=":")
plt.ylabel("h(x) = v_max - v")
plt.xlabel("Time (s)")
plt.legend()
plt.title("CBF h(x) Safety Value (Solid: Estimated, Dashed: Real)")


plt.tight_layout()
plt.show()

# OURS (TVRCBF)의 Safety Tube
plt.figure(figsize=(10, 6))
h_hat_our = results["our"]["h_hat"]
P_list_our = results["our"]["P"]
a = np.array([0.0, -1.0])
aTPa_sqrt = np.array([np.sqrt(a @ P @ a.T) for P in P_list_our])
tube_radius = np.sqrt(beta) * aTPa_sqrt

plt.plot(time, h_hat_our, label="h_hat (OURS)", color=colors["our"])
plt.fill_between(time, h_hat_our - tube_radius, h_hat_our + tube_radius, alpha=0.25, color=colors["our"], label="Safety Tube (OURS)")
plt.plot(time, results["our"]["h_real"], color="black", linestyle="--", label="h_real (OURS)")
plt.axhline(0, color="gray", linestyle=":")
plt.title("CBF Safety with Tube (OURS)")
plt.xlabel("Time (s)")
plt.ylabel("h(x)")
plt.legend()
plt.tight_layout()
plt.show()