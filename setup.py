import sys

import setuptools
from setuptools import setup

__version__ = "0.0"

if sys.version_info < (3, 10):
    sys.exit("lcs requires Python 3.10 or later.")

name = "lcs"

version = __version__

authors = "Nicholas Landry and Will Thompson"

author_email = "nicholas.landry@uvm.edu"


def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if not l.startswith("#")]
    return requires


install_requires = parse_requirements_file("requirements.txt")

license = "3-Clause BSD license"

setup(
    name=name,
    packages=setuptools.find_packages(),
    version=version,
    author=authors,
    author_email=author_email,
    install_requires=install_requires,
)
