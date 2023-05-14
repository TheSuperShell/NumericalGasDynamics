import numpy as np


# Вектор консервативных параметров
def q(rho_j, u_j, E_j) -> np.array:
    return np.array((rho_j, rho_j * u_j, rho_j * E_j))


# Обратно из вектора в параметры
def reverse_q(q_j: np.array, gamma=1.4):
    rho_j = q_j[0]
    u_j = q_j[1] / rho_j
    E_j = q_j[2] / rho_j
    p_j = (gamma - 1) * rho_j * (E_j - u_j * u_j / 2)
    return p_j, rho_j, u_j, E_j


# Вектор потоков
def f(p_j, rho_j, u_j, E_j):
    return np.array((rho_j * u_j, rho_j * u_j * u_j + p_j, (rho_j * E_j + p_j) * u_j))


def a(p_j, rho_j, gamma=1.4):
    return np.sqrt(gamma * p_j / rho_j)


def energy(p_j, rho_j, u_j, gamma=1.4):
    return p_j / ((gamma - 1) * rho_j) + u_j * u_j / 2
