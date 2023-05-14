import numpy as np

from data import GasData2D, Parameters2D


def hll(f_l, f_r, q_l, q_r, a_l, a_r, u_l, u_r) -> np.array:
	"""
	HLL method of solving arbitrary discontinuity
	:param f_l: flow
	:param f_r: flow with shift to right
	:param q_l: q
	:param q_r: q with shift to right
	:param a_l: a
	:param a_r: a with shift to right
	:param u_l: u
	:param u_r: u with shift to right
	:return: flow vector
	"""
	s_minus = np.min((u_r - a_r, u_l - a_l), axis=0)[:, None]
	s_plus = np.max((u_l + a_l, u_r + a_r), axis=0)[:, None]

	f_hll = (s_plus * f_l - s_minus * f_r + s_plus * s_minus * (q_r - q_l)) / (s_plus - s_minus)

	f_res = np.where(s_minus > 0, f_l, f_hll)
	f_res = np.where(s_plus < 0, f_r, f_res)
	return f_res


def first_order(params: GasData2D) -> np.array:
	f_vec = params.parameters.get_f()
	f_l = f_vec[:-1]
	f_r = f_vec[1:]
	q_vec = params.parameters.get_q()
	q_l = q_vec[:-1]
	q_r = q_vec[1:]
	a_vec = params.parameters.get_a()
	a_l = a_vec[:-1]
	a_r = a_vec[1:]
	u_l = params.parameters.u[:-1]
	u_r = params.parameters.u[1:]
	return hll(f_l, f_r, q_l, q_r, a_l, a_r, u_l, u_r)


def second_order(params: GasData2D) -> np.array:
	params_l, params_r = get_edges(params)
	f_l = params_l.get_f()[:-1]
	f_r = params_r.get_f()[1:]
	q_l = params_l.get_q()[:-1]
	q_r = params_r.get_q()[1:]
	a_l = params_l.get_a()[:-1]
	a_r = params_r.get_a()[1:]
	u_l = params_l.u[:-1]
	u_r = params_r.u[1:]
	return hll(f_l, f_r, q_l, q_r, a_l, a_r, u_l, u_r)


def get_edges(params: GasData2D):
	"""
	calculates parameters based on the slope on the edges of each volume
	:param params: task parameters
	:return: left edge parameters and right edge parameters
	"""
	def get_diff(q: np.array, dx: float) -> np.array:
		q_l = q[:-2]
		q_m = q[1:-1]
		q_r = q[2:]

		a = (q_r - q_m) / dx
		b = (q_m - q_l) / dx
		return 0.5 * (np.sign(a) + np.sign(b)) * np.min((np.abs(a), np.abs(b)), axis=0)

	dp = get_diff(params.parameters.p, params.delta_x)
	drho = get_diff(params.parameters.rho, params.delta_x)
	du = get_diff(params.parameters.u, params.delta_x)

	params_l = Parameters2D(params.Nx).copy_parameters(params.parameters)
	params_r = Parameters2D(params.Nx).copy_parameters(params.parameters)
	params_l.p[1:-1] += dp * params.delta_x * 0.5
	params_r.p[1:-1] -= dp * params.delta_x * 0.5
	params_l.rho[1:-1] += drho * params.delta_x * 0.5
	params_r.rho[1: -1] -= drho * params.delta_x * 0.5
	params_l.u[1: -1] += du * params.delta_x * 0.5
	params_r.u[1: -1] -= du * params.delta_x * 0.5
	params_l.update_energy()
	params_r.update_energy()
	return params_l, params_r
