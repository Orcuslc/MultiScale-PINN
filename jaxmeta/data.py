import numpy as np
import jax.numpy as jnp
import jax
import os

def tensor_grid(x, order = None):
	"""build tensor grid for multiple parameters
	
	Arguments:
		x {tuple or list of np.array} -- parameters
		order {tuple or list of int, optional} -- order of output dimensions
	
	Returns:
		grid {np.ndarray} -- tensor grids

	When `order` is None, it is equivalent to `range(len(x))`.

	Example:
		>>> tensor_grid(([1, 2], [3, 4], [5, 6, 7]))
		>>> np.array([[1, 3, 5],
					[1, 3, 6],
					[1, 3, 7],
					[1, 4, 5],
					[1, 4, 6],
					[1, 4, 7],
					[2, 3, 5],
					[2, 3, 6],
					[2, 3, 7],
					[2, 4, 5],
					[2, 4, 6],
					[2, 4, 7]])

		>>> tensor_grid([[1, 2], [3, 4], [5, 6]], order = (2, 0, 1))
		>>> np.array([[1, 3, 5],
					[1, 4, 5],
					[2, 3, 5],
					[2, 4, 5],
					[1, 3, 6],
					[1, 4, 6],
					[2, 3, 6],
					[2, 4, 6]])
	"""
	if order is not None:
		assert len(order) == len(x), "Length of order should be the same with length of x"
		x = [x[i] for i in order]
	grid = np.vstack(np.meshgrid(*x, indexing = 'ij')).reshape((len(x), -1)).T
	if order is not None:
		grid[:, order] = grid[:, list(range(len(x)))]
	return grid

def cast(x, dtype = jnp.float32):
	return list(map(lambda x: jnp.array(x, dtype = dtype), x))

def save_params(path, x):
	if not os.path.exists(os.path.dirname(path)):
		os.makedirs(os.path.dirname(path))
	np.save(path, np.asarray(x))

def load_params(path):
	x = np.load(path, allow_pickle = True)
	return [[jnp.asarray(arr) for arr in Arr] for Arr in x]

@jax.jit
def normalize(x, domain):
	return 2*(x - (domain[0, :]+domain[1, :]))/(domain[1, :] - domain[0, :])