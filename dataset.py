import jax
import jax.numpy as jnp
from jax import random

class Batch_Generator:
	def __init__(self, key, dataset, batch_size, static_dataset = None):
		self.key = key
		self.dataset = dataset
		self.batch_size = batch_size
		self.static_dataset = static_dataset
		self.index = jnp.arange(dataset[0].shape[0])
		self.pointer = 0
		self._shuffle()

	def _shuffle(self):
		self.key, subkey = random.split(self.key)
		self.index = random.permutation(subkey, jnp.arange(self.dataset[0].shape[0]))

	def __iter__(self):
		return self

	def __next__(self):
		if self.pointer >= len(self.index):
			self._shuffle()
			self.pointer = 0
		self.pointer += self.batch_size
		index_ = self.index[self.pointer-self.batch_size:self.pointer]
		next_batch = [d[index_, :] for d in self.dataset]
		if self.static_dataset is None:
			return next_batch
		else:
			return next_batch, self.static_dataset


class Time_Marching_Batch_Generator:
	def __init__(self, key, domain, batch_size, iterations, static_dataset):
		self.key = key
		self.domain = domain
		self.batch_size = batch_size
		self.iterations = iterations
		self._iteration = 0
		self.static_dataset = static_dataset

	def _sample(self, domain):
		self._iteration += 1
		self.key, subkey = random.split(self.key)
		return random.uniform(key, (self.batch_size, 1), jnp.float32, *domain)

	def __iter__(self):
		return self

	def __next__(self):
		if self._iteration > self.iterations:
			self._iteration = self.iterations
		domain = [self.domain[0], self.domain[0] + (self.domain[1]-self.domain[0])/iterations*self._iteration]
		t_sample = self._sample(domain)
		return t_sample, self.static_dataset
