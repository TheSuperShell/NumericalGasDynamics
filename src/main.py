from simulation.data import GasData2D, Parameters2D
from simulation.godunov import n_steps
from simulation.arbitary_break import second_order
from simulation.energy_input import HyperbolicEnergyInput

if __name__ == '__main__':

	init_conditions = Parameters2D(1002)
	init_conditions.const_parameter('p', 26600)
	init_conditions.const_parameter('rho', 0.33)
	init_conditions.update_energy()

	data_1 = GasData2D('discharge', 1000, 0.05, 0.30)
	data_1.set_initial_parameters(init_conditions)
	data_1.delete_parameters()

	energy_input = HyperbolicEnergyInput(0, 0.000002, 100000, 0, 0.002)

	n_steps(data_1, 1000, False, solver=second_order, rodionov=True, energy_input=energy_input)
	data_1.time_base()
