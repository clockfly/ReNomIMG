import os
import sys
import re
import shutil
import pathlib
import numpy
from setuptools import setup, find_packages, Extension
import distutils.command.build
from Cython.Build import cythonize
from distutils.extension import Extension


if sys.version_info < (3, 4):
    raise RuntimeError('renom_img requires Python3')

DIR = str(pathlib.Path(__file__).resolve().parent)

requires = [
    "bs4", "bottle", "glob2", "lxml", "Pillow",
    "PyYAML", "watchdog", "xmltodict", "tqdm"
]


entry_points = {
    'console_scripts': [
        'renom_img = renom_img.server.server:main',
    ]
}


class BuildNPM(distutils.command.build.build):
    """Custom build command."""

    def run(self):
        shutil.rmtree(os.path.join(DIR, 'renom_img/server/.build'), ignore_errors=True)
        curdir = os.getcwd()
        try:
            jsdir = os.path.join(DIR, 'js')

            # skip if js directory not exists.
            if os.path.isdir(jsdir):
                os.chdir(jsdir)
                ret = os.system('npm install')
                if ret:
                    raise RuntimeError('Failed to install npm modules')

                ret = os.system('npm run build')
                if ret:
                    raise RuntimeError('Failed to build npm modules')

        finally:
            os.chdir(curdir)

        super().run()


setup(
    name="renom_img",
    version="1.0.9",
    entry_points=entry_points,
    packages=['renom_img'],
    install_requires=requires,
    include_package_data=True,
    zip_safe=True,
    cmdclass={
        'build': BuildNPM,
    },
    ext_modules=cythonize([
                          "renom_img/api/utility/*.pyx",
                          "renom_img/api/utility/evaluate/*.pyx",
                          ],
                          include_path=[numpy.get_include()])
)
