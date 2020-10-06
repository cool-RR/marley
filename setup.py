# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.
import setuptools
import re


def read_file(filename):
    with open(filename) as file:
        return file.read()

version = re.search("__version__ = '([0-9.]*)'",
                    read_file('grid_royale/__init__.py')).group(1)

setuptools.setup(
    name='grid_royale',
    version=version,
    author='Ram Rachum',
    author_email='ram@rachum.com',
    description='GridRoyale - A life simulation for exploring social dynamics',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/cool-RR/grid_royale',
    packages=setuptools.find_packages(exclude=['tests*']),
    install_requires=read_file('requirements.txt'),
    include_package_data=True, # For including the frontend files
    extras_require={
        'tests': {
            'pytest',
        },
    },
    entry_points={
        'console_scripts': [
            'grid_royale = grid_royale:grid_royale'
        ],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Artificial Life',
    ],
)
