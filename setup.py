#!/usr/bin/python
#-*-coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='airfoiltools',
    version='0.1.0',
    description='Airfoil analysis tools - Bezier curves, profile geometry, XFoil integration',
    author='Nervures',
    author_email='be@nervures.com',
    license='LGPL-3.0',
    packages=find_packages(),
    package_data={
        'airfoiltools': ['*.cfg'],
    },
    install_requires=[
        'numpy>=1.16',
    ],
    python_requires='>=2.7',
)
