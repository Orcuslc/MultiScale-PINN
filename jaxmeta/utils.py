import time
import jax.numpy as jnp
import numpy as np
from typing import List

def get_time():
	return time.strftime("%Y/%m/%d, %H:%M:%S", time.localtime())

def apply_to_nested_list(x, fn):
	if isinstance(x, list):
		return [apply_to_nested_list(xi, fn) for xi in x]
	else:
		return fn(x)

def flatten_list(x: List) -> List:
	if isinstance(x, list):
		if len(x) == 1:
			return flatten_list(x[0])
		else:
			return flatten_list(x[0]) + flatten_list(x[1:])
	else:
		return [x]

def unflatten_list(x: List, structure: List) -> List:
	def _iter(x: List, structure: List, index: int):
		if isinstance(structure, list):
			left, index = _iter(x, structure[0], index)
			if len(structure) == 1:
				return [left], index
			else:
				right, index = _iter(x, structure[1:], index)
				return [left] + right, index
		else:
			return x[index], index+1
	return _iter(x, structure, 0)[0]

def unflatten_to_shape(x: jnp.array, shapes: List) -> List:
	flatten_shapes = flatten_list(shapes)
	arrays = []
	index = 0
	for shape in flatten_shapes:
		index_end = index + np.prod(shape)
		arrays.append(x[index:index_end])
		index = index_end
	return unflatten_list(arrays, shapes)