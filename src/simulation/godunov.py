from data import GasData2D, Parameters2D
from energy_input import EnergyInput
from arbitary_break import first_order, get_edges
import numpy as np
from time import time


def solve_numerically(q_k, f_k, dt, dx, g_k, x, t, energy_input: EnergyInput) -> np.array:
	f_l = f_k[:-1]
	f_r = f_k[1:]
	return get_q_k1(q_k, f_l, f_r, dt, dx, g_k, x, t, energy_input)


def get_q_k1(q_k, f_l, f_r, dt, dx, g_k, x, t, energy_input: EnergyInput) -> np.array:
	q_k1 = q_k.copy()
	q_k1[1: -1] = q_k[1: -1] - dt / dx * (f_r - f_l) + g_k[1: -1] * dt
	if energy_input:
		N = q_k1.shape[0]
		de = np.zeros((N, 3))
		de[:, 2] = energy_input.get_energy_input(x, t)
		q_k1 += de * dt
	return q_k1


def step(params: GasData2D, dt: float, solver, rodionov: bool, energy_input: EnergyInput):
	N = params.Nx
	q_k = params.parameters.get_q()
	g_k = np.zeros((N, 3))
	if params.cylindrical:
		g_k = params.parameters.get_g(params.x)

	if rodionov:
		edge_l, edge_r = get_edges(params)
		f_r = edge_l.get_f()[1:-1]
		f_l = edge_r.get_f()[1:-1]
		q_k1 = get_q_k1(q_k, f_l, f_r, dt,params.delta_x, g_k, params.x, params.t, energy_input)
	else:
		f_k = solver(params)
		q_k1 = solve_numerically(q_k, f_k, dt, params.delta_x, g_k, params.x, params.t, energy_input)

	if params.cylindrical:
		g_k1 = Parameters2D(N).reverse_q(q_k1).get_g(params.x)
		g_k = 0.5 * (g_k + g_k1)

	if rodionov:
		q_k1 = 0.5 * (q_k1 + q_k)
		new_params = Parameters2D(N).reverse_q(q_k1)
		params.set_parameters(new_params, 0, False)
		f_k = solver(params)

	if params.cylindrical or rodionov:
		q_k1 = solve_numerically(q_k, f_k, dt, params.delta_x, g_k, params.x, params.t, energy_input)

	new_params = Parameters2D(N).reverse_q(q_k1)
	params.set_parameters(new_params, dt)


def n_steps(
		params: GasData2D,
		max_nt: int,
		fixed_dt=False,
		solver=first_order,
		rodionov=False,
		energy_input: EnergyInput = None,
		verbose=100
):
	start = time()
	while params.Nt < max_nt:
		if params.Nt % verbose == 0:
			print(f'Loading: {params.Nt / max_nt:.0%}')
		dt = params.delta_t * params.k if fixed_dt else params.get_dt()
		step(params, dt, solver, rodionov, energy_input)
	print(f'Finished in {time() - start:.4f} seconds')


def t_steps(
		params: GasData2D,
		max_t: float,
		fixed_dt: bool = False,
		solver=first_order,
		rodionov: bool = False,
		energy_input: EnergyInput = None,
		verbose = 100
):
	start = time()
	timer = verbose
	while params.t < max_t:
		if timer == 0:
			timer = verbose
			print(f't = {params.t}. Left {params.t / max_t:.0%}')
		timer -= 1
		dt = params.delta_t * params.k if fixed_dt else params.get_dt()
		step(params, dt, solver, rodionov, energy_input)
	print(f'Finished in {time() - start:.4f} seconds')
