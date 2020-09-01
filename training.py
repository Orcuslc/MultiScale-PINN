import jax 
import jax.numpy as jnp
from functools import partial
from jaxmeta.utils import get_time
from jaxmeta.data import save_params, save_data
import sys, os

class Agent:
	def __init__(self, params, loss_fn, evaluate_fn, save_path):
		self._params = params
		self.loss_fn = loss_fn
		self.evaluate_fn = evaluate_fn
		self.save_path = save_path

	def compile(self, optimizer, lr):
		opt_init, self.opt_update, self.get_params = optimizer(lr)
		self.opt_state = opt_init(self._params)
		self.hist = {"iter": [], "loss": []}
		self._iteration = 0

	@partial(jax.jit, static_argnums = (0,))
	def _step(self, i, opt_state, batch):
		params = self.get_params(opt_state)
		grad = jax.grad(self.loss_fn, 0)(params, batch)
		opt_state = self.opt_update(i, grad, opt_state)
		return opt_state

	@property
	def params(self):
		return self.get_params(self.opt_state)

	def train(self, iterations, batch_fn, evaluate_batch_fn, print_every, save_every, loss_names, log_file = None):
		for i in range(1, iterations+1):
			self._iteration += 1
			self.opt_state = self._step(self._iteration, self.opt_state, batch_fn(self._iteration))
			if i % print_every == 0:
				losses = self.evaluate_fn(self.params, evaluate_batch_fn(self._iteration))
				print("{}, Iteration: {}, Train".format(get_time(), self._iteration) + \
					','.join([" {}: {:.4e}".format(name, loss) for name, loss in zip(loss_names, losses)]), file = sys.stdout if log_file is None else log_file)
				self.hist["iter"].append(i)
				self.hist["loss"].append(losses)
			if i % save_every == 0:
				save_path = "{}/iteration_{}/params.npy".format(self.save_path, self._iteration)
				save_params(save_path, self.params)
		save_data("{}/hist.pkl".format(self.save_path), self.hist)
		

