#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import codecs
import os

from setuptools import find_packages, setup

# Basic information
NAME = "Het Shah"
DESCRIPTION = "A Pytorch Library to help extend all Knowledge Distillation works"
AUTHOR = "Het Shah"
EMAIL = "divhet163@gmail.com"
LICENSE = "MIT"
REPOSITORY = "https://github.com/SforAiDl/KD_Lib"
PACKAGE = "KD_Lib"
with open("README.rst", "r") as f:
    LONG_DESCRIPTION = f.read()

# Define the keywords
KEYWORDS = ["Knowledge Distillation", "Pruning", "Quantization", "pytorch", "machine learning", "deep learning"]
setup_requirements = ['pytest-runner']

test_requirements = ['pytest', 'pytest-cov']

# Define the classifiers
# See https://pypi.python.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS = [
	'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
]

# Important Paths
PROJECT = os.path.abspath(os.path.dirname(__file__))
REQUIRE_PATH = "requirements.txt"
VERSION_PATH = os.path.join(PACKAGE, "version.py")
PKG_DESCRIBE = "README.rst"

# Directories to ignore in find_packages
EXCLUDES = ()


# helper functions
def read(*parts):
    """
    returns contents of file
    """
    with codecs.open(os.path.join(PROJECT, *parts), "rb", "utf-8") as file:
        return file.read()


def get_requires(path=REQUIRE_PATH):
    """
    generates requirements from file path given as REQUIRE_PATH
    """
    for line in read(path).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


# Define the configuration
CONFIG = {
    "name": NAME,
    "version": 'version='0.0.20'',
    "description": DESCRIPTION,
    "long_description": LONG_DESCRIPTION,
    "long_description_content_type": "text/markdown",
    "classifiers": CLASSIFIERS,
    "keywords": KEYWORDS,
    "license": LICENSE,
    "author": AUTHOR,
    "author_email": EMAIL,
    "url": REPOSITORY,
    "project_urls": {"Source": REPOSITORY},
    "packages": find_packages(where=PROJECT, exclude=EXCLUDES),
    "install_requires": list(get_requires()),
    "python_requires": ">=3.6",
    "setup_requires": setup_requirements,
    "test_suite": "tests",
    "tests_require": test_requirements,
}

if __name__ == "__main__":
    setup(**CONFIG)
