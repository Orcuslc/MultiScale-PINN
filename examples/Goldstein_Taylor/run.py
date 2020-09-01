import jax
import jax.numpy as jnp

import config
from train import train

n_data = [
	{
		"i": 10**i,
		"b": 10**i,
		"cx": 4**i,
		"ct": 4**i,
	}
	for i in range(1, 5)
]
batch_size = [
	{
		"dirichlet": 3*10**i,
		"collocation": 4**(2*i),
	}
	for i in range(1, 5)
]

for i, n_ in enumerate(n_data):
	for j, b_ in enumerate(batch_size):
		config.NAME = "d{}_c{}".format(i, j)
		config.n_data = n_
		config.batch_size = b_
		config.log_file = open("{}.log".format(config.NAME), "w")
		train(config)
