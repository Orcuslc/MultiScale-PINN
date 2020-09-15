import jax 
import jax.numpy as jnp
from jax.scipy.optimize import minimize

from functools import partial
import sys, os
from typing import List

from jaxmeta.utils import get_time, apply_to_nested_list, flatten_list, unflatten_list, unflatten_to_shape
from jaxmeta.data import save_params, save_data

class Agent:
	def __init__(self, params, loss_fn, evaluate_fn, save_path):
		self._params = params
		self.loss_fn = loss_fn
		self.evaluate_fn = evaluate_fn
		self.save_path = save_path

	def compile(self, optimizer, lr, params_init = None):	
		if params_init is None or not hasattr(self, "opt_state"):
			self.opt_init, self.opt_update, self.get_params = optimizer(lr)
			self.opt_state = self.opt_init(self._params)
			self.hist = {"iter": [], "loss": []}
			self._iteration = 0
		else:
			self.opt_state = self.opt_init(params_init)

	@partial(jax.jit, static_argnums = (0,))
	def _step(self, i, opt_state, batch):
		params = self.get_params(opt_state)
		grad = jax.grad(self.loss_fn, 0)(params, batch)
		opt_state = self.opt_update(i, grad, opt_state)
		return opt_state

	@property
	def params(self) -> List[List[jnp.array]]:
		return self.get_params(self.opt_state)

	def train(self, iterations, batch_fn, evaluate_batch_fn, print_every, save_every, loss_names, log_file = None) -> None:
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

	def train_bfgs(self, n_batches, batch_fn, options, loss_names, log_file = None, scale = 1.0):
		param_shapes = apply_to_nested_list(self.params, lambda x: x.shape)
		flatten = flatten_list(self.params)
		flatten_params = jnp.hstack([x.reshape(-1,) for x in flatten])
		
		@jax.jit
		def loss_fn_bfgs(params, batch):
			params_ = unflatten_to_shape(params, param_shapes)
			return self.loss_fn(params_, batch)*scale

		for i in range(n_batches):
			batch = batch_fn(i)
			loss_fn_batch = jax.jit(partial(loss_fn_bfgs, batch = batch))
			opt_results = minimize(loss_fn_batch, 
								   flatten_params, 
								   method = "bfgs",
								   tol = 1e-7,
								   options = options)
			print("Success: {},\n Status: {},\n Message: {},\n nfev: {},\n njev: {},\n nit: {}".format(opt_results.success, opt_results.status, opt_results.message, opt_results.nfev, opt_results.njev, opt_results.nit))

			flatten_params = opt_results.x
			losses = self.evaluate_fn(unflatten_to_shape(flatten_params, param_shapes), batch)
			print("{}, Batch: {}, BFGS".format(get_time(), i) + \
					','.join([" {}: {:.4e}".format(name, loss) for name, loss in zip(loss_names, losses)]), file = sys.stdout if log_file is None else log_file)

		return unflatten_to_shape(flatten_params, param_shapes)
		

