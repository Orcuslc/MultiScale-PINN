import jax
import jax.numpy as jnp

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training import Agent
from loss import model, params, loss_fn, evaluate_fn
from data import generate_dataset, generate_batch_fn
# import config

def train(config):
	datasets = generate_dataset(config.n_data["i"], config.n_data["b"], config.n_data["cx"], config.n_data["ct"])
	batch_fn, evaluate_batch_fn = generate_batch_fn(config.key, config.batch_size, *datasets, config.weights)

	agent = Agent(params, loss_fn, evaluate_fn, "models/{}".format(config.NAME))
	agent.compile(config.optimizer, config.lr)
	agent.train(config.iterations, batch_fn, evaluate_batch_fn, config.print_every, config.save_every, config.loss_names, config.log_file)