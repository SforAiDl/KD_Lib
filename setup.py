#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import codecs
import os

from setuptools import find_packages, setup

# Basic information
with open("README.rst", "r") as f:
    LONG_DESCRIPTION = f.read()

# Define the keywords
KEYWORDS = ["Knowledge Distillation", "Pruning", "Quantization", "pytorch", "machine learning", "deep learning"]
REQUIRE_PATH = "requirements.txt"
PROJECT = os.path.abspath(os.path.dirname(__file__))
setup_requirements = ['pytest-runner']

test_requirements = ['pytest', 'pytest-cov']

requirements = [
'pip==19.3.1',
'transformers==4.6.1',
'sacremoses',
'tokenizers==0.10.1',
'huggingface-hub==0.0.8',
'torchtext==0.9.1',
'bumpversion==0.5.3',
'wheel==0.32.1',
'watchdog==0.9.0',
'flake8==3.5.0',
'tox==3.5.2',
'coverage==4.5.1',
'Sphinx==1.8.1',
'twine==1.12.1',
'pytest==3.8.2',
'pytest-runner==4.2',
'pytest-cov==2.6.1',
'matplotlib==3.2.1',
'torch==1.8.1',
'torchvision==0.9.1',
'tensorboard==2.2.1',
'contextlib2==0.6.0.post1',
'pandas==1.0.1',
'tqdm==4.42.1',
'numpy==1.18.1',
'sphinx-rtd-theme==0.5.0',
]


if __name__ == "__main__":
    setup(
	author="Het Shah",
	author_email='divhet163@gmail.com',
    	classifiers=[
        	'Development Status :: 2 - Pre-Alpha',
       		'Intended Audience :: Developers',
        	'License :: OSI Approved :: MIT License',
        	'Natural Language :: English',
        	'Programming Language :: Python :: 3.6',
        	'Programming Language :: Python :: 3.7',
    	],
    	description="A Pytorch Library to help extend all Knowledge Distillation works",
    	install_requires=requirements,
    	license="MIT license",
    	long_description=LONG_DESCRIPTION,
    	include_package_data=True,
    	keywords=KEYWORDS,
    	name='KD_Lib',
    	packages=find_packages(where=PROJECT),
    	setup_requires=setup_requirements,
    	test_suite="tests",
    	tests_require=test_requirements,
    	url="https://github.com/SforAiDL/KD_Lib",
    	version='0.0.29',
    	zip_safe=False,
)
