from data import GasData2D, Parameters2D
from energy_input import EnergyInput
from arbitary_break import first_order
import numpy as np


def solve_numerically(q_k, f_k, dt, dx, g_k, t, energy_input: EnergyInput) -> np.array:
	q_k1 = q_k.copy()
	N = q_k1.shape[0]
	for j in range(1, N-1):
		f_r = f_k[j]
		f_l = f_k[j-1]
		q_k1[j] = q_k[j] - dt / dx * (f_r - f_l) + g_k[j-1] * dt
		if energy_input:
			r = (j - 0.5) * dx
			epsilon = np.array((0, 0, energy_input.get_energy_input(r, t)))
			q_k1[j] += epsilon * dt
	return q_k1


def step(params: GasData2D, dt: float, solver, rodionov: bool, energy_input: EnergyInput):
	N = params.Nx
	q_k = params.parameters.get_q()
	g_k = np.zeros((N, 3))
	if params.cylindrical:
		g_k = params.parameters.get_g(params.x)

	f_k = solver(params)
	q_k1 = solve_numerically(q_k, f_k, dt, params.delta_x, g_k, params.t, energy_input)

	if params.cylindrical:
		g_k1 = Parameters2D(N).reverse_q(q_k1).get_g(params.x)
		g_k = 0.5 * (g_k + g_k1)

	if rodionov:
		q_k1 = 0.5 * (q_k1 + q_k)
		new_params = Parameters2D(N).reverse_q(q_k1)
		params.set_parameters(new_params, 0, False)
		f_k = solver(params)

	if params.cylindrical or rodionov:
		q_k1 = solve_numerically(q_k, f_k, dt, params.delta_x, g_k, params.t, energy_input)

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
	while params.Nt < max_nt:
		if params.Nt % verbose == 0:
			print(f'Loading: {params.Nt / max_nt:.1%}')
		dt = params.delta_t * params.k if fixed_dt else params.get_dt()
		step(params, dt, solver, rodionov, energy_input)

	params.draw_graphs()
	params.time_base()
