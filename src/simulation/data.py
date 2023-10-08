import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


class Parameters2D:
	def __init__(self, number_of_x: int, gamma: float = 1.4):
		self.p = np.zeros(number_of_x)
		self.rho = np.ones(number_of_x)
		self.u = np.zeros(number_of_x)
		self.E = np.zeros(number_of_x)
		self.gamma = gamma

	def get_parameter_by_name(self, param_name) -> np.array:
		if param_name == 'p':
			return self.p
		if param_name == 'rho':
			return self.rho
		if param_name == 'u':
			return self.u
		if param_name == 'E':
			return self.E
		return None

	def set_parameter_by_name(self, param_name: str, array: np.array, update_e: bool = False) -> None:
		if param_name == 'p':
			self.p = array
		elif param_name == 'rho':
			self.rho = array
		elif param_name == 'u':
			self.u = array
		elif param_name == 'E':
			self.E = array
			return
		if update_e:
			self.update_energy()

	def get_all(self):
		return self.p, self.rho, self.u, self.E

	def copy(self):
		return Parameters2D(self.p.shape[0], self.gamma).copy_parameters(self)

	def copy_parameters(self, params):
		self.p = params.p.copy()
		self.rho = params.rho.copy()
		self.u = params.u.copy()
		return self

	def const_parameter(self, param_name: str, value: float) -> None:
		new_param = np.ones(len(self.p)) * value
		self.set_parameter_by_name(param_name, new_param)

	def update_energy(self):
		self.E = self.p / ((self.gamma - 1) * self.rho) + self.u * self.u / 2
		return self

	def get_a(self) -> np.array:
		return np.sqrt(self.gamma * self.p / self.rho)

	def get_q(self) -> np.array:
		return np.array((self.rho, self.rho * self.u, self.rho * self.E)).T

	def reverse_q(self, q):
		self.rho = q[:, 0]
		self.u = q[:, 1] / self.rho
		self.E = q[:, 2] / self.rho
		self.p = (self.gamma - 1) * self.rho * (self.E - self.u * self.u / 2)
		return self

	def get_f(self) -> np.array:
		return np.array((
			self.rho * self.u,
			self.rho * self.u * self.u + self.p,
			(self.rho * self.E + self.p) * self.u
		)).T

	def get_g(self, r: np.array) -> np.array:
		g1 = -self.rho * self.u / r
		g2 = -self.rho * self.u * self.u / r
		g3 = -(self.rho * self.E + self.p) * self.u / r
		return np.array((g1, g2, g3)).T

	def left_condition(self, cond_type: str):
		if cond_type == 'closed':
			self.p[0] = self.p[1]
			self.rho[0] = self.rho[1]
			self.u[0] = -self.u[1]
			self.E[0] = self.E[1]
			return
		if cond_type == 'open':
			self.p[0] = self.p[1]
			self.rho[0] = self.rho[1]
			self.u[0] = self.u[1]
			self.E[0] = self.E[1]

	def right_condition(self, cond_type: str):
		if cond_type == 'closed':
			self.p[-1] = self.p[-2]
			self.rho[-1] = self.rho[-2]
			self.u[-1] = -self.u[-2]
			self.E[-1] = self.E[-2]
			return
		if cond_type == 'open':
			self.p[-1] = self.p[-2]
			self.rho[-1] = self.rho[-2]
			self.u[-1] = self.u[-2]
			self.E[-1] = self.E[-2]


class GasData2D:
	PARAMETERS = {
		'p': 0,
		'rho': 1,
		'u': 2,
		'E': 3
	}

	def __init__(
			self,
			name: str,
			number_of_x: int = None,
			length: float = None,
			k: float = None,
			boundary_conditions: tuple = ('closed', 'closed'),
			cylindrical: bool = False
	):

		self.name = name
		self.file_path = './save/' + self.name

		if not os.path.exists(self.file_path):
			os.mkdir(self.file_path)

		settings_file = os.path.join(self.file_path, '_settings.json')
		if os.path.isfile(settings_file):
			with open(settings_file, 'r') as f:
				data_json = f.readline()
			data_dict = json.loads(data_json)
			number_of_x = data_dict['Nx']
			length = data_dict['L']
			k = data_dict['k']
			cylindrical = data_dict['cylindrical']
			boundary_conditions = data_dict['boundary_conditions']
			print('Successfully loaded data!')
		else:
			if not (number_of_x and length and k):
				raise ValueError("File doesn't exist, you must enter new parameters")
			data_dict = {
				'name': name,
				'Nx': number_of_x,
				'L': length,
				'k': k,
				'boundary_conditions': boundary_conditions,
				'cylindrical': cylindrical
			}
			data_json = json.dumps(data_dict)
			with open(settings_file, 'w') as f:
				f.write(data_json)

		self.k = k
		self.Nx = number_of_x + 2
		self.L = length
		self.cylindrical = cylindrical
		self.boundary_conditions = boundary_conditions

		self.delta_x = length / number_of_x
		self.delta_t = 0.411 * self.delta_x
		self.x = np.linspace(-self.delta_x * 0.5, self.L + self.delta_x * 0.5, self.Nx, dtype=np.float32)

		self.time_path = os.path.join(self.file_path, 'time.txt')
		t = 0
		number_of_t = 0
		if os.path.isfile(self.time_path):
			time_array = self.get_time_array()
			if len(time_array) > 0:
				t = float(time_array[-1].strip())
				number_of_t = len(time_array)
		else:
			with open(self.time_path, 'w') as f:
				print('', end='', file=f)
		self.t = t
		self.Nt = number_of_t
		self.parameters = Parameters2D(self.Nx)
		self.init_parameters = Parameters2D(self.Nx)

		for param_name in GasData2D.PARAMETERS:
			param_path = os.path.join(self.file_path, param_name + '.param')
			if os.path.isfile(param_path):
				param_array = self.get_parameter_array(param_name)
				if param_array.shape[0] != self.Nt:
					raise ValueError('param array shape doesnt match time array shape!')
				if param_array.shape[0] != 0:
					self.parameters.set_parameter_by_name(param_name, param_array[-1])
					self.init_parameters.set_parameter_by_name(param_name, param_array[0])
			else:
				with open(param_path, 'wb') as f:
					f.write(b'')

	def get_time_array(self):
		if not os.path.isfile(self.time_path):
			raise FileNotFoundError('File with time array doesnt exist!')

		with open(self.time_path, 'r') as f:
			return f.readlines()

	def get_parameter_array(self, param_name):
		path = os.path.join(self.file_path, param_name + '.param')
		if not os.path.isfile(path):
			raise FileNotFoundError(f'Parameter file {param_name} doesnt exist!')

		with open(path, 'rb') as f:
			data = f.read()
		return np.frombuffer(data).reshape((self.Nt, self.Nx))

	def get_parameter_at_k(self, param_name, k: int):
		return self.get_parameter_array(param_name)[k]

	def get_current_parameter(self, param_name):
		return self.parameters.get_parameter_by_name(param_name)

	def save_parameters(self):
		for param_name in GasData2D.PARAMETERS:
			path = os.path.join(self.file_path, param_name + '.param')
			if not os.path.isfile(path):
				raise FileNotFoundError(f'File for parameter {param_name} doesnt exist!')
			parameter: np.array = self.parameters.get_parameter_by_name(param_name)
			with open(path, 'ab') as f:
				f.write(parameter.tobytes())

	def delete_parameters(self):
		self.parameters = self.init_parameters
		self.t = 0
		self.Nt = 1
		with open(self.time_path, 'w') as f:
			print('0', file=f)
		for param_name in GasData2D.PARAMETERS:
			path = os.path.join(self.file_path, param_name + '.param')
			parameter = self.init_parameters.get_parameter_by_name(param_name)
			with open(path, 'wb') as f:
				f.write(parameter.tobytes())

	def set_parameters(self, new_parameters: Parameters2D, dt: float, save: bool = True):
		self.apply_boundary_conditions(new_parameters)
		self.parameters = new_parameters
		if save:
			self.Nt += 1
			self.t += dt
			with open(self.time_path, 'a') as f:
				print(self.t, file=f)
				self.save_parameters()

	def set_initial_parameters(self, init_parameters: Parameters2D, show_p: bool = False):
		self.apply_boundary_conditions(init_parameters)
		self.init_parameters = init_parameters
		if show_p:
			plt.plot(self.x, self.init_parameters.p)
			plt.title(r'Начальные условия для $p$')
			plt.show()
		if self.Nt > 0:
			warnings.warn('Initial conditions are already set')
			return
		self.set_parameters(init_parameters, 0)

	def apply_boundary_conditions(self, parameters: Parameters2D):
		parameters.left_condition(self.boundary_conditions[0])
		parameters.right_condition(self.boundary_conditions[1])

	def get_dt(self) -> float:
		a = self.parameters.get_a()
		delta_t = np.zeros(self.Nx - 2)
		for j in range(1, self.Nx - 1):
			delta_t[j - 1] = self.delta_x / np.max(
				(np.abs(self.parameters.u[j] + a[j]), np.abs(self.parameters.u[j] - a[j])))
		return np.min(delta_t) * self.k

	def draw_graphs(self, k=-1):
		p_k = self.get_parameter_at_k('p', k)
		rho_k = self.get_parameter_at_k('rho', k)
		u_k = self.get_parameter_at_k('u', k)
		e_k = self.get_parameter_at_k('E', k)

		plt.figure(figsize=(7, 4 * 4))

		plt.subplot(4, 1, 1)
		plt.plot(self.x[1:self.Nx - 1], p_k[1:self.Nx - 1])
		plt.xlabel("$x$, м")
		plt.ylabel("$p$, Па")

		plt.subplot(4, 1, 2)
		plt.plot(self.x[1:self.Nx - 1], rho_k[1:self.Nx - 1])
		plt.xlabel("$x$, м")
		plt.ylabel(r"$\rho$, кг/$м^3$")

		plt.subplot(4, 1, 3)
		plt.plot(self.x[1:self.Nx - 1], u_k[1:self.Nx - 1])
		plt.xlabel("$x$, м")
		plt.ylabel(r"$u$, м/с")

		plt.subplot(4, 1, 4)
		plt.plot(self.x[1:self.Nx - 1], e_k[1:self.Nx - 1])
		plt.xlabel("$x$, м")
		plt.ylabel(r"$E$, Дж/кг")
		plt.show()

	def time_base(self, xlims=(0, 1)):
		rho = self.get_parameter_array('rho')
		nxlims = (int(np.floor(xlims[0] * self.Nx)) + 1, int(np.floor(xlims[1] * self.Nx)))
		im = plt.imshow(
			rho[-1:0:-1, nxlims[0]:nxlims[1]],
			cmap='plasma_r',
			aspect='auto',
			extent=[xlims[0] * self.L * 1000, xlims[1] * self.L * 1000, 0, float(self.get_time_array()[-1]) * 1000000]
		)
		cbar = plt.colorbar(im)
		cbar.ax.set_ylabel(r'$\rho$, $кг/м^3$')
		plt.xlabel('r, мм')
		plt.ylabel('t, мкс')
		plt.title('Развертка по времени')
		plt.show()

	def save_gif(self, param_name, skip=1):
		param = self.get_parameter_array(param_name)
		t_array = self.get_time_array()

		fig = plt.figure()
		ax = plt.axes(xlim=(0, self.L), ylim=(param.min(), param.max()))
		plt.title(f'{param_name}')
		plt.xlabel('r, м')
		plt.ylabel(f'{param_name}')
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		label = ax.text(
			0.95,
			0.90,
			't = 0 мкс',
			transform=ax.transAxes,
			fontsize=14,
			horizontalalignment='right',
			bbox=props)
		line, = ax.plot([], [], lw=1.5)

		def init():
			line.set_data([], [])
			return line,

		def step(i):
			k = i * skip
			x = self.x[1:-1]
			y = param[k, 1:-1]
			line.set_data(x, y)
			t_k = float(t_array[k])
			new_text = f't = {t_k * 1000000:.2f} мкс'
			label.set_text(new_text)
			return line,

		anim = FuncAnimation(fig, step, init_func=init, frames=self.Nt // skip - 1, interval=1, blit=True)
		save_path = os.path.join(self.file_path, 'animation_' + param_name + '.gif')
		anim.save(save_path)
		plt.close()
