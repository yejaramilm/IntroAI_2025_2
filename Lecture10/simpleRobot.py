import numpy as np

# Definimos estados y acciones
STATES = ["A", "B"]
ACTIONS = ["stay", "move"]
SIDX = {s: i for i, s in enumerate(STATES)}
AIDX = {a: i for i, a in enumerate(ACTIONS)}

# P[s, a, s'] = probabilidad de transitar de s a s' tomando acción a
P = np.zeros((len(STATES), len(ACTIONS), len(STATES)))
R = np.zeros((len(STATES), len(ACTIONS), len(STATES)))

# Estado A
P[SIDX["A"], AIDX["stay"], SIDX["A"]] = 1.0
R[SIDX["A"], AIDX["stay"], SIDX["A"]] = 0

P[SIDX["A"], AIDX["move"], SIDX["B"]] = 1.0
R[SIDX["A"], AIDX["move"], SIDX["B"]] = +1

# Estado B
P[SIDX["B"], AIDX["stay"], SIDX["B"]] = 1.0
R[SIDX["B"], AIDX["stay"], SIDX["B"]] = 0

P[SIDX["B"], AIDX["move"], SIDX["A"]] = 1.0
R[SIDX["B"], AIDX["move"], SIDX["A"]] = 0

# Política simple: siempre "move"
policy = {"A": "move", "B": "move"}

def evaluate_policy(P, R, policy, gamma=0.9, theta=1e-6):
    """Evaluación de política determinista con iteración de valores"""
    V = np.zeros(len(STATES))
    stable = False
    while not stable:
        delta = 0
        for s in STATES:
            i = SIDX[s]
            a = AIDX[policy[s]]
            v = V[i]
            V[i] = sum(P[i, a, j] * (R[i, a, j] + gamma * V[j]) 
                       for j in range(len(STATES)))
            delta = max(delta, abs(v - V[i]))
        if delta < theta:
            stable = True
    return {s: round(V[SIDX[s]], 3) for s in STATES}

# Ejecutar
values = evaluate_policy(P, R, policy, gamma=0.9)
print("Valores de la política 'always move':", values)
