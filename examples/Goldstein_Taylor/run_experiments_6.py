import jax
import jax.numpy as jnp

import config
from train import train

config.PATH = "experiments_6"

n_ib = [{"i": 10*2**i, "b": 10*2**i} for i in [4]]
n_c = [{"cx": 5*2**i, "ct": 5*2**i} for i in range(0, 6)]

config.layers = [2] + [64]*4 + [2]
config.iterations = 500000

if __name__ == "__main__":
	for i, n_ib_ in enumerate(n_ib):
		for j, n_c_ in enumerate(n_c):
			name = "ib_{}_c_{}".format(i, j)
			config.NAME = "{}/{}".format(config.PATH, name)
			config.n_data = {
				"i": n_ib_["i"], "b": n_ib_["b"], "cx": n_c_["cx"], "ct": n_c_["ct"]
			}
			config.batch_size = {
				"dirichlet": n_ib_["i"] + 2*n_ib_["b"],
				"collocation": n_c_["cx"]*n_c_["ct"]
			}
			config.log_file = open("{}.log".format(name), "w")
			train(config)
