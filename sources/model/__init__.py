#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Package airfoiltools : analyse aerodynamique 2D de profils.

Courbes de Bezier, geometrie de profils, integration XFoil.

Usage::

    from airfoiltools import Profil, Bezier

    p = Profil.from_naca('2412')
    p.approximate_bezier(degree=8)

@author: Nervures
@date: 2026-02
"""

from .bezier import Bezier
from .profil import Profil
from .simulation import Simulation, SimulationResults
from .analyse import Analyse
from .pipeline import FoilAnalysisPipeline, register_solver
from .base import AbstractPreprocessor, AbstractSimulator, AbstractPostprocessor
from .foilconfig import load_config, load_defaults, merge_params
