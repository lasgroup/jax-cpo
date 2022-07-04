#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'jax',
    'dm-haiku',
    'optax',
    'jmp',
    'numpy',
    'ruamel.yaml',
    'tensorboardX',
    'tensorflow',
    'tensorflow-probability',
    'moviepy',
    'gym'
]

setup(
    name='jax-cpo',
    version='0.0.0',
    packages=find_packages(),
    python_requires='>3.8',
    include_package_data=True,
    install_requires=required)
