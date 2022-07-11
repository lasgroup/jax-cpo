#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'jax', 'dm-haiku', 'optax', 'jmp', 'numpy', 'ruamel.yaml', 'tensorboardX',
    'tensorflow', 'tensorflow-probability', 'moviepy', 'gym'
]

extras = {'dev': ['pytest>=4.4.0', 'mujoco-py', 'safe-adaptation-gym']}

setup(
    name='jax-cpo',
    version='0.0.0',
    packages=find_packages(),
    python_requires='>3.8',
    include_package_data=True,
    install_requires=required,
    dependency_links=[
        'http://github.com/lasgroup/safe-adaptation-gym/tarball/main#egg=safe-adaptation-gym'
    ],
    extras_require=extras)
