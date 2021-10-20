# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.
import setuptools
import re
import sys
import pathlib



def read_file(filename):
    with open(filename) as file:
        return file.read()
    
def was_arezzo_built():
    arezzo_folder = pathlib.Path(__file__).parent / 'marley' / 'arezzo' / 'arezzo'
    arezzo_src_folder = arezzo_folder / 'src'
    arezzo_dist_folder = arezzo_folder / 'dist'
    all_arezzo_src_paths = tuple(arezzo_src_folder.rglob('*'))
    all_arezzo_dist_paths = tuple(arezzo_dist_folder.rglob('*'))
    newest_arezzo_src_creation_time = max(path.stat().st_ctime for path in all_arezzo_src_paths)
    oldest_arezzo_dist_modification_time = min(path.stat().st_mtime for path in all_arezzo_dist_paths)
    return oldest_arezzo_dist_modification_time > newest_arezzo_src_creation_time

if any(('dist' in arg) for arg in sys.argv):
    if not was_arezzo_built():
        raise Exception(
            "Arezzo source code was modified, it needs to be rebuilt before you can build Marley. "
            "Rebuild it by going into the `marley/arezzo/arezzo` folder and running `npm run "
            "build`"
        )


version = re.search("__version__ = '([0-9.]*)'",
                    read_file('marley/__init__.py')).group(1)

setuptools.setup(
    name='marley',
    version=version,
    author='Ram Rachum',
    author_email='ram@rachum.com',
    description='Marley - A framework for multi-agent reinforcment learning',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/cool-RR/marley',
    packages=setuptools.find_packages(exclude=['tests*']),
    install_requires=read_file('requirements.txt'),
    include_package_data=True, # For including the frontend files
    python_requires='>=3.8',
    extras_require={
        'tests': {
            'pytest',
            'pytest-xdist',
            'pytest-html',
        },
    },
    entry_points={
        'console_scripts': [
            'marley = marley.commanding:marley'
        ],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
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
