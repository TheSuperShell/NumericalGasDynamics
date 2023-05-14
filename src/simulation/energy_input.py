from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EnergyInput:
	t0: float
	delta_t: float
	delta_e: float
	x0: float
	x1: float

	def get_energy_input(self, x: np.array, t: np.array) -> np.array:
		raise NotImplementedError()

	def _apply_mask(self, x: np.array, t: np.array, func) -> np.array:
		result = np.where(x < self.x0, 0, func(t))
		result = np.where(x > self.x1, 0, result)
		result = np.where(t < self.t0, 0, result)
		result = np.where(t > self.t0 + self.delta_t, 0, result)
		return result

	def draw_energy_input(self, n: int):
		t = np.linspace(self.t0, self.t0 + self.delta_t, n, dtype=np.float32)
		e = self.get_energy_input((self.x0 + self.x1) / 2, t)
		plt.plot(t, e)
		plt.title('Энерговклад от времени')
		plt.grid()
		plt.show()


class HyperbolicEnergyInput(EnergyInput):
	def get_energy_input(self, x: np.array, t: np.array) -> np.array:
		e_max = 3.0 / 2.0 * self.delta_e / self.delta_t
		a = -4 * e_max / (self.delta_t * self.delta_t)
		b = -a * self.delta_t

		def parabola(t_inp):
			return a * (t_inp - self.t0) ** 2 + b * (t_inp - self.t0)
		return self._apply_mask(x, t, parabola)
