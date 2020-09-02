import jax
import jax.numpy as jnp

import config
from train import train

n_ib = [{"i": 10*4**i, "b": 10*4**i} for i in range(1, 7)]
n_c = [{"cx": 10*2**i, "ct": 10*2**i} for i in range(1, 7)]

if __name__ == "__main__":
	for i, n_ib_ in enumerate(n_ib):
		for j, n_c_ in enumerate(n_c):
			config.NAME = "ib_{}_c_{}".format(i, j)
			config.n_data = {
				"i": n_ib_["i"], "b": n_ib_["b"], "cx": n_c_["cx"], "ct": n_c_["ct"]
			}
			config.batch_size = {
				"dirichlet": n_ib_["i"] + 2*n_ib_["b"],
				"collocation": n_c_["cx"]*n_c_["ct"]
			}
			config.log_file = open("{}.log".format(config.NAME), "w")
			train(config)
