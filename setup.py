import os
from setuptools import setup, find_packages

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
version = '1.0.0'

# Get the documentation
with open(os.path.join(cwd, 'README.md'), "r") as fh:
    long_description = fh.read()

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: Other/Proprietary License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python :: 3.11",
]

setup(
    name             = "crn_paper",
    author           = "Daniel J. Klein, Romesh Abeysuriya, Robyn M. Stuart, and Cliff C. Kerr",
    author_email     = "idm@gatesfoundation.org",
    description      = 'Code associated with the manuscript, "Taming Randomness in Agent-Based Models using Common Random Numbers"',
    url              = 'http://idmod.org',
    keywords         = ["Agent-based model", "Common random numbers", "Variance reduction", "Starsim"],
    install_requires = [
        "starsim",
    ],

    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
)
