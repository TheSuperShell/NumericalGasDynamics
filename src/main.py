from simulation.data import GasData2D, Parameters2D
from simulation.godunov import n_steps, t_steps
from simulation.arbitary_break import second_order
from simulation.energy_input import HyperbolicEnergyInput
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

	data_1 = GasData2D('discharge')
	data_1.delete_parameters()

	energy_input = HyperbolicEnergyInput(0, 0.000002, 100000, 0, 0.002)

	t_steps(data_1, 0.000050, False, solver=second_order, rodionov=True, energy_input=energy_input)
	data_1.time_base()
