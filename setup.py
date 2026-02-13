#!/usr/bin/python
#-*-coding: utf-8 -*-

from setuptools import setup

setup(
    name='airfoiltools',
    version='0.1.0',
    description='Airfoil analysis tools - Bezier curves, profile geometry, XFoil integration',
    author='Nervures',
    author_email='be@nervures.com',
    license='LGPL-3.0',
    package_dir={
        'airfoiltools': 'sources/model',
        'airfoiltools_gui': 'sources/gui',
    },
    packages=['airfoiltools', 'airfoiltools_gui'],
    package_data={
        'airfoiltools': ['*.cfg'],
    },
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'matplotlib>=3.5',
    ],
    extras_require={
        'gui': ['PySide6>=6.5'],
    },
    python_requires='>=3.8',
)
