import functions as fnc
import numpy as np

from data import GasData2D


def hll(p0, rho0, u0, e0, p1, rho1, u1, e1):

	f0 = fnc.f(p0, rho0, u0, e0)
	f1 = fnc.f(p1, rho1, u1, e1)

	a0 = fnc.a(p0, rho0)
	a1 = fnc.a(p1, rho1)

	s_minus = np.min((u1 - a1, u0 - a0))
	if s_minus > 0:
		return f0
	s_plus = np.max((u1 + a1, u0 + a0))
	if s_plus < 0:
		return f1

	q0 = fnc.q(rho0, u0, e0)
	q1 = fnc.q(rho1, u1, e1)

	f_hll = s_plus * f0 - s_minus * f1 + s_plus * s_minus * (q1 - q0)
	f_hll /= s_plus - s_minus

	return f_hll


def first_order(params: GasData2D) -> np.array:
	n = params.Nx
	p_k, rho_k, u_k, E_k = params.parameters.get_all()
	f_k = np.zeros((n-1, 3))
	for j in range(n-1):
		f_k[j] = hll(p_k[j], rho_k[j], u_k[j], E_k[j], p_k[j+1], rho_k[j+1], u_k[j+1], E_k[j+1])
	return f_k


def second_order(params: GasData2D) -> np.array:

	def diff(q_l, q_j, q_r, dx):
		a = (q_r - q_j) / dx
		b = (q_j - q_l) / dx
		return 0.5 * (np.sign(a) + np.sign(b)) * np.min((np.abs(a), np.abs(b)))

	N = params.Nx
	p_k, rho_k, u_k, E_k = params.parameters.get_all()
	f_k = np.zeros((N - 1, 3))
	d_k = np.zeros((N, 3))
	for j in range(1, N - 1):
		p_l, rho_l, u_l = p_k[j - 1], rho_k[j - 1], u_k[j - 1]
		p_i, rho_i, u_i = p_k[j], rho_k[j], u_k[j]
		p_r, rho_r, u_r = p_k[j + 1], rho_k[j + 1], u_k[j + 1]

		dp = diff(p_l, p_i, p_r, params.delta_x)
		drho = diff(rho_l, rho_i, rho_r, params.delta_x)
		du = diff(u_l, u_i, u_r, params.delta_x)
		d_k[j] = np.array((dp, drho, du))

		if j != 1:
			p_l += d_k[j - 1, 0] * params.delta_x * 0.5
			rho_l += + d_k[j - 1, 1] * params.delta_x * 0.5
			u_l += + d_k[j - 1, 2] * params.delta_x * 0.5
		p_r = p_i - d_k[j, 0] * params.delta_x * 0.5
		rho_r = rho_i - d_k[j, 1] * params.delta_x * 0.5
		u_r = u_i - d_k[j, 2] * params.delta_x * 0.5

		E_l, E_r = fnc.energy(p_l, rho_l, u_l), fnc.energy(p_r, rho_r, u_r)
		f_k[j - 1] = hll(p_l, rho_l, u_l, E_l, p_r, rho_r, u_r, E_r)

		if j == N - 2:
			p_l = p_i + d_k[j, 0] * params.delta_x * 0.5
			rho_l = rho_i + d_k[j, 1] * params.delta_x * 0.5
			u_l = u_i + d_k[j, 2] * params.delta_x * 0.5
			p_r, rho_r, u_r = p_k[j + 1], rho_k[j + 1], u_k[j + 1]

			E_l, E_r = fnc.energy(p_l, rho_l, u_l), fnc.energy(p_r, rho_r, u_r)
			f_k[j] = hll(p_l, rho_l, u_l, E_l, p_r, rho_r, u_r, E_r)
	return f_k

