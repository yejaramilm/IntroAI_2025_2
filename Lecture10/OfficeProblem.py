# -*- coding: utf-8 -*-
"""
MDP/MRP de ejemplo basado en el grafo: Chat, Coffee, Computer, Home.
- Probabilidades en NEGRO del diagrama (transiciones).
- Recompensas en ROJO del diagrama (por transición s -> s').
Incluye:
  * Verificaciones de estocasticidad
  * Cálculo de r(s) (recompensa inmediata esperada)
  * Evaluación de valor V para MRP (solución exacta y por iteración)
  * Simulación de episodios
  * Estimación de P a partir de episodios (conteo y normalización)
  * Infraestructura opcional para acciones (MDP genérico)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple, Optional
import random


# ----------------------------
# 1) Definición del entorno
# ----------------------------
STATES = ["Chat", "Coffee", "Computer", "Home"]
SIDX = {s: i for i, s in enumerate(STATES)}

# P[s, s'] = prob de ir de s a s'
P = np.zeros((len(STATES), len(STATES)), dtype=float)

# Chat
P[SIDX["Chat"], SIDX["Chat"]]     = 0.5
P[SIDX["Chat"], SIDX["Computer"]] = 0.3
P[SIDX["Chat"], SIDX["Coffee"]]   = 0.2

# Coffee
P[SIDX["Coffee"], SIDX["Chat"]]     = 0.7
P[SIDX["Coffee"], SIDX["Computer"]] = 0.2
P[SIDX["Coffee"], SIDX["Coffee"]]   = 0.1

# Computer
P[SIDX["Computer"], SIDX["Computer"]] = 0.5
P[SIDX["Computer"], SIDX["Chat"]]     = 0.1
P[SIDX["Computer"], SIDX["Coffee"]]   = 0.2
P[SIDX["Computer"], SIDX["Home"]]     = 0.2

# Home
P[SIDX["Home"], SIDX["Home"]]   = 0.6
P[SIDX["Home"], SIDX["Coffee"]] = 0.4

# R[s, s'] = recompensa al transitar s -> s'
R = np.full((len(STATES), len(STATES)), np.nan, dtype=float)
R[SIDX["Chat"],     SIDX["Chat"]]     = -1
R[SIDX["Chat"],     SIDX["Computer"]] =  2
R[SIDX["Chat"],     SIDX["Coffee"]]   =  1

R[SIDX["Coffee"],   SIDX["Chat"]]     =  2
R[SIDX["Coffee"],   SIDX["Computer"]] =  3
R[SIDX["Coffee"],   SIDX["Coffee"]]   =  1

R[SIDX["Computer"], SIDX["Computer"]] =  5
R[SIDX["Computer"], SIDX["Chat"]]     = -3
R[SIDX["Computer"], SIDX["Coffee"]]   =  1
R[SIDX["Computer"], SIDX["Home"]]     =  2

R[SIDX["Home"],     SIDX["Home"]]     =  1
R[SIDX["Home"],     SIDX["Coffee"]]   =  1


# -----------------------------------------
# 2) Utilidades y verificaciones básicas
# -----------------------------------------
def check_stochastic(P: np.ndarray, tol: float = 1e-12) -> None:
    """Verifica que cada fila de P sume ~1 y que no haya probs negativas."""
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        raise ValueError(f"Algunas filas no suman 1: {row_sums}")
    if (P < -tol).any():
        raise ValueError("Se detectaron probabilidades negativas.")

def expected_immediate_rewards(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    r(s) = E[R_{t+1} | S_t = s] = sum_{s'} P(s->s') * R(s->s')
    (si hay transiciones imposibles, R puede traer NaN; por eso usamos nansum)
    """
    return np.nansum(P * R, axis=1)


# -----------------------------------------
# 3) Evaluación de valores (MRP)
# -----------------------------------------
def solve_values_linear(P: np.ndarray, r: np.ndarray, gamma: float) -> np.ndarray:
    """
    Solución exacta: (I - gamma P) V = r  =>  V = (I - gamma P)^{-1} r
    Requiere |gamma|<1 si no hay estados absorbentes que garanticen solución.
    """
    I = np.eye(P.shape[0])
    A = I - gamma * P
    return np.linalg.solve(A, r)

def evaluate_values_iter(P: np.ndarray, r: np.ndarray, gamma: float,
                         theta: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """
    Iteración de valores para MRP (convergencia cuando gamma<1).
    V_{k+1} = r + gamma P V_k
    """
    V = np.zeros_like(r)
    for _ in range(max_iter):
        V_new = r + gamma * P.dot(V)
        if np.max(np.abs(V_new - V)) < theta:
            return V_new
        V = V_new
    return V  # por si se supera max_iter


# -----------------------------------------
# 4) Simulación de episodios desde P y R
# -----------------------------------------
def sample_next_state(s: int, P: np.ndarray) -> int:
    """Muestra el siguiente estado ~ P[s, :]"""
    return np.random.choice(len(STATES), p=P[s])

def sample_reward(s: int, s_next: int, R: np.ndarray) -> float:
    """Recompensa de la transición s -> s_next (según matriz R)."""
    return float(R[s, s_next])

def simulate_episode(start_state: str,
                     P: np.ndarray,
                     R: np.ndarray,
                     n_steps: int,
                     seed: Optional[int] = None) -> List[Tuple[str, str, float]]:
    """
    Simula una trayectoria: [(s, s', r), ...] de longitud n_steps.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    s = SIDX[start_state]
    traj = []
    for _ in range(n_steps):
        s_next = sample_next_state(s, P)
        r = sample_reward(s, s_next, R)
        traj.append((STATES[s], STATES[s_next], r))
        s = s_next
    return traj


# -----------------------------------------
# 5) Estimar P desde episodios (conteo)
# -----------------------------------------
def estimate_transition_matrix(episodes: List[List[Tuple[str, str, float]]]) -> np.ndarray:
    """
    Estima P por conteo de transiciones s->s' a partir de episodios simulados/observados.
    """
    counts = np.zeros_like(P)
    for ep in episodes:
        for (s, s_next, _r) in ep:
            counts[SIDX[s], SIDX[s_next]] += 1
    # Normalización por filas
    row_sums = counts.sum(axis=1, keepdims=True)
    # Evitar división por cero si alguna fila no tuvo visitas:
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


# -----------------------------------------
# 6) Infraestructura opcional para MDP
#    (acciones). Aquí usamos una sola acción
#    para reproducir EXACTAMENTE el diagrama.
# -----------------------------------------
@dataclass
class MDP:
    states: List[str]
    actions: Dict[str, List[str]]  # acciones disponibles por estado
    P: Dict[Tuple[str, str], Dict[str, float]]  # (s,a) -> {s': prob}
    R: Dict[Tuple[str, str, str], float]        # (s,a,s') -> reward

    def policy_evaluation(self, policy: Dict[str, str], gamma: float,
                          theta: float = 1e-10, max_iter: int = 10_000) -> Dict[str, float]:
        """
        Evaluación de una política determinista: V^pi(s)
        V(s) = sum_{s'} P(s,a->s') [ R(s,a,s') + gamma * V(s') ]
        """
        V = {s: 0.0 for s in self.states}
        for _ in range(max_iter):
            delta = 0.0
            for s in self.states:
                a = policy[s]
                pv = 0.0
                for s2, p in self.P[(s, a)].items():
                    r = self.R[(s, a, s2)]
                    pv += p * (r + gamma * V[s2])
                delta = max(delta, abs(pv - V[s]))
                V[s] = pv
            if delta < theta:
                break
        return V


# -----------------------------------------
# 7) DEMO / Ejecutar
# -----------------------------------------
if __name__ == "__main__":
    # 7.1 Verificaciones
    check_stochastic(P)
    r = expected_immediate_rewards(P, R)
    print("r(s) esperado inmediato por estado:", dict(zip(STATES, r.round(4))))
    # -> {'Chat': 0.3, 'Coffee': 2.1, 'Computer': 2.8, 'Home': 1.0}

    # 7.2 Valores por solución exacta (MRP)
    for gamma in [0.0, 0.5, 0.9, 0.99]:
        V_exact = solve_values_linear(P, r, gamma)
        print(f"gamma={gamma:>4}:",
              dict(zip(STATES, np.round(V_exact, 4))))
    # Observa que conforme gamma->1, los valores crecen (divergen si gamma=1)

    # 7.3 Valores por iteración (deben coincidir con la solución exacta)
    V_iter = evaluate_values_iter(P, r, gamma=0.9)
    print("V por iteración (gamma=0.9):", dict(zip(STATES, V_iter.round(4))))

    # 7.4 Simulación de un episodio (para "ver" la dinámica)
    ep = simulate_episode(start_state="Home", P=P, R=R, n_steps=12, seed=42)
    print("\nEpisodio simulado (s -> s' | r):")
    for s, s2, rr in ep:
        print(f"{s:>8} -> {s2:<9} | r={rr:+.1f}")

    # 7.5 Estimar P desde muchos episodios (conteo y normalización)
    episodes = [simulate_episode("Home", P, R, n_steps=30) for _ in range(500)]
    P_hat = estimate_transition_matrix(episodes)
    print("\nComparación P real vs P estimada (filas ~ estados):")
    for i, s in enumerate(STATES):
        print(f"{s:>8}  real={P[i].round(2)}   est={P_hat[i].round(2)}")

    # 7.6 (Opcional) Construir un MDP con una sola acción para usar policy_evaluation
    actions = {s: ["do"] for s in STATES}
    P_mdp = {}
    R_mdp = {}
    for s in STATES:
        P_row = {STATES[j]: float(P[SIDX[s], j]) for j in range(len(STATES)) if P[SIDX[s], j] > 0}
        P_mdp[(s, "do")] = P_row
        for s2, p in P_row.items():
            R_mdp[(s, "do", s2)] = float(R[SIDX[s], SIDX[s2]])  # recompensa por transición

    mdp = MDP(states=STATES, actions=actions, P=P_mdp, R=R_mdp)
    policy = {s: "do" for s in STATES}  # única acción
    V_pi = mdp.policy_evaluation(policy, gamma=0.9)
    print("\nV^pi por evaluación de política (MDP 1-acción, gamma=0.9):",
          {k: round(v, 4) for k, v in V_pi.items()})
