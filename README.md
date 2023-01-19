# Constrained Policy Optimization with JAX
Constrained Policy Optimization is a safe reinforcement learning algorithm that solves constrained Markov decision processes to ensure safety. Our implementation is a port of the [original OpenAI implementation](https://github.com/openai/safety-starter-agents) to JAX.

## Install
Create a self-contained environment (via [conda](https://docs.conda.io/en/latest/) or [virtualenv](https://virtualenv.pypa.io/en/latest/)); for instance:
```
conda create -n <jax-cpo> python=3.8
conda activate jax-cpo
```
Install requirements:
```
pip3 install -r requirements.txt
```

## Usage

### Via [`Trainer`](https://github.com/lasgroup/jax-cpo/blob/main/jax_cpo/trainer.py) class
This is the easier entry point for running experiments. A usage example, and tests, are provided [here](https://github.com/lasgroup/jax-cpo/blob/main/tests/test_cpo.py).

### With your own training loop
If you just want to use our implementation with a different training/evaluation setup, you can directly use the [`CPO`](https://github.com/lasgroup/jax-cpo/blob/main/jax_cpo/cpo.py) class. The only required interface is via the `__call__(observation: np.ndarray, train: bool) -> np.array` function. The function implements the following:
* Observes the state (provided by the environment), put it in an episodic buffer for the next policy update.
* At each timestep use the current policy to return an action.
* Whenever the `train` flag is true, and the buffer is full, a policy update is triggered.

Consult [`configs.yaml`](https://github.com/lasgroup/jax-cpo/blob/main/jax_cpo/configs.yaml) for hyper-parameters.

