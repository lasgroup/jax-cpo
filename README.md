# Constrained Policy Optimization with JAX
Constrained Policy Optimization is a safe reinforcement learning algorithm that solves constrained Markov decision processes to ensure safety. Our implementation is a port of the [original OpenAI implementation](https://github.com/openai/safety-starter-agents) to JAX.

## Install
First, make sure to have a python 3.10.12 installed.
### Using Poetry
```
poetry install
```
Check out the additional (optional) installation groups in `pyproject.toml` for additional functionality.
### Without Poetry
You have two options, cloning the repository (for example, for local development and hacking) or just install it as it is, directly from github.
1. Clone: `git clone https://github.com/lasgroup/jax-cpo.git`, then `cd jax-cpo` and `pip install -e .`; or
2. `pip install git+https://git@github.com/lasgroup/jax-cpo`


## Usage

### Via [`Trainer`](https://github.com/lasgroup/jax-cpo/blob/main/jax_cpo/rl/trainer.py) class
This is the easier entry point for running experiments. A usage example [here](https://github.com/lasgroup/jax-cpo/blob/main/main.py).

### With your own training loop
If you just want to use our implementation with a different training/evaluation setup, you can directly use the [`CPO`](https://github.com/lasgroup/jax-cpo/blob/main/jax_cpo/cpo.py) class. The only required interface is via the `__call__(observation: np.ndarray, train: bool) -> np.array` function. The function implements the following:
* Observes the state (provided by the environment), put it in an episodic buffer for the next policy update.
* At each timestep use the current policy to return an action.
* Whenever the `train` flag is true, and the buffer is full, a policy update is triggered.

Consult [`configs.yaml`](https://github.com/lasgroup/jax-cpo/blob/main/jax_cpo/configs.yaml) for hyper-parameters.