#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "matplotlib>=3.2.1",
    "torch>=1.5.0",
    "torchvision>=0.6.0",
    "tensorboard>=2.2.1",
    "torchtext>=0.6.0",
    "transformers>=0.6.0",
    "pandas>=1.0.1",
    "tqdm>=4.42.1",
    "numpy>=1.18.1",
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest", "pytest-cov"]

setup(
    author="Het Shah",
    author_email="divhet163@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="A Pytorch Library to help extend all Knowledge Distillation works",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="KD_Lib",
    name="KD_Lib",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/SforAiDL/KD_Lib",
    version="version='0.0.9'",
    zip_safe=False,
)
