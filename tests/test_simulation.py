#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Tests pour les classes Simulation et SimulationResults.

Tests autonomes (sans XFoil), utilisant des resultats synthetiques.

Lance :
    python test_simulation.py

@author: Nervures
@date: 2026-02
"""

import os
import sys
import unittest

import numpy as np

# Ajouter le repertoire airfoiltools/ au path
_here = os.path.dirname(os.path.abspath(__file__))
_pkg = os.path.normpath(os.path.join(_here, '..', 'airfoiltools'))
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)

from simulation import Simulation, SimulationResults
from profil import Profil


def make_fake_results(re_list=None, n_alpha=25):
    u"""Cree un dict de resultats synthetiques, identique au format pipeline.

    :param re_list: liste de Reynolds (defaut: [500000])
    :param n_alpha: nombre de points alpha
    :returns: dict compatible pipeline.run()
    """
    if re_list is None:
        re_list = [500000.0]

    polars = {}
    cp = {}
    for re_val in re_list:
        alpha = np.linspace(-2, 10, n_alpha)
        cl = 0.11 * alpha
        cd = 0.008 + 0.001 * alpha**2
        cdp = cd * 0.6
        cm = np.full_like(alpha, -0.02)
        xtr_top = np.clip(0.5 - 0.03 * alpha, 0.01, 1.0)
        xtr_bot = np.clip(0.5 + 0.05 * alpha, 0.01, 1.0)

        polars[re_val] = {
            'alpha': alpha,
            'CL': cl,
            'CD': cd,
            'CDp': cdp,
            'CM': cm,
            'Top_Xtr': xtr_top,
            'Bot_Xtr': xtr_bot
        }

        # Cp synthetique pour alpha=0 et alpha=5
        cp[re_val] = {}
        for a in [0.0, 5.0]:
            x = np.linspace(0, 1, 20)
            cp_vals = 1.0 - 4.0 * x * (1.0 - x)
            cp[re_val][a] = np.column_stack([x, cp_vals])

    return {
        'polars': polars,
        'cp': cp,
        'bl': {},
        'warnings': []
    }


# ======================================================================
#  Tests SimulationResults
# ======================================================================

class TestSimulationResults(unittest.TestCase):
    u"""Tests de la classe SimulationResults."""

    def setUp(self):
        raw = make_fake_results([500000.0, 1000000.0])
        self.results = SimulationResults.from_dict(raw)

    def test_from_dict(self):
        u"""Creation depuis dict pipeline."""
        self.assertIsInstance(self.results, SimulationResults)
        self.assertTrue(self.results.has_polars)

    def test_re_list(self):
        u"""Liste triee des Reynolds."""
        re_list = self.results.re_list
        self.assertEqual(len(re_list), 2)
        self.assertAlmostEqual(re_list[0], 500000.0)
        self.assertAlmostEqual(re_list[1], 1000000.0)

    def test_get_polar(self):
        u"""Acces polaire par Re."""
        polar = self.results.get_polar(500000.0)
        self.assertIsNotNone(polar)
        self.assertIn('alpha', polar)
        self.assertIn('CL', polar)
        self.assertIn('CD', polar)
        self.assertEqual(len(polar['alpha']), 25)

    def test_get_polar_default(self):
        u"""get_polar(None) retourne le premier Re."""
        polar = self.results.get_polar()
        self.assertIsNotNone(polar)

    def test_get_polar_missing(self):
        u"""get_polar avec Re inexistant -> None."""
        self.assertIsNone(self.results.get_polar(999999.0))

    def test_get_cp(self):
        u"""Acces Cp par (Re, alpha)."""
        cp = self.results.get_cp(500000.0, 5.0)
        self.assertIsNotNone(cp)
        self.assertEqual(cp.shape, (20, 2))

    def test_get_cp_missing(self):
        u"""Cp avec alpha inexistant -> None."""
        self.assertIsNone(self.results.get_cp(500000.0, 99.0))

    def test_alpha_range(self):
        u"""Plage alpha."""
        rng = self.results.alpha_range(500000.0)
        self.assertIsNotNone(rng)
        self.assertAlmostEqual(rng[0], -2.0, places=1)
        self.assertAlmostEqual(rng[1], 10.0, places=1)

    def test_cl_max(self):
        u"""CL_max et alpha correspondant."""
        result = self.results.cl_max(500000.0)
        self.assertIsNotNone(result)
        cl_max, alpha_cl_max = result
        # CL = 0.11 * alpha, donc CL_max a alpha=10
        self.assertAlmostEqual(alpha_cl_max, 10.0, places=1)
        self.assertAlmostEqual(cl_max, 1.1, delta=0.1)

    def test_finesse_max(self):
        u"""Finesse max et alpha correspondant."""
        result = self.results.finesse_max(500000.0)
        self.assertIsNotNone(result)
        fmax, alpha_fmax = result
        self.assertGreater(fmax, 0)

    def test_n_converged(self):
        u"""Comptage des points converges."""
        # 2 Re x 25 pts = 50
        self.assertEqual(self.results.n_converged, 50)

    def test_empty_results(self):
        u"""Resultats vides."""
        r = SimulationResults()
        self.assertFalse(r.has_polars)
        self.assertFalse(r.has_cp)
        self.assertEqual(r.n_converged, 0)
        self.assertEqual(r.re_list, [])
        self.assertIsNone(r.get_polar())
        self.assertIsNone(r.cl_max())
        self.assertIsNone(r.finesse_max())

    def test_warnings(self):
        u"""Acces aux warnings."""
        raw = make_fake_results()
        raw['warnings'] = ['warn1', 'warn2']
        r = SimulationResults.from_dict(raw)
        self.assertEqual(len(r.warnings), 2)

    def test_repr(self):
        u"""__repr__ lisible."""
        r = repr(self.results)
        self.assertIn('Re', r)
        self.assertIn('converges', r)


# ======================================================================
#  Tests Simulation
# ======================================================================

class TestSimulation(unittest.TestCase):
    u"""Tests de la classe Simulation."""

    def setUp(self):
        self.profil = Profil.from_naca('2412')

    def test_creation(self):
        u"""Constructeur, etat initial 'idle'."""
        sim = Simulation(self.profil)
        self.assertEqual(sim.state, 'idle')
        self.assertIsNone(sim.results)
        self.assertIsNone(sim.error)
        self.assertFalse(sim.is_done)
        self.assertFalse(sim.is_failed)
        self.assertFalse(sim.has_results)

    def test_profil_reference(self):
        u"""simulation.profil est bien le profil passe."""
        sim = Simulation(self.profil)
        self.assertIs(sim.profil, self.profil)

    def test_solver_default(self):
        u"""Solveur par defaut = xfoil."""
        sim = Simulation(self.profil)
        self.assertEqual(sim.solver, 'xfoil')

    def test_params_merge(self):
        u"""Parametres utilisateur fusionnes avec les defauts."""
        sim = Simulation(self.profil, params={'RE': 1000000, 'NCRIT': 5})
        self.assertEqual(sim.params['RE'], 1000000)
        self.assertEqual(sim.params['NCRIT'], 5)
        # Les defauts doivent etre presents
        self.assertIn('ALPHA_MIN', sim.params)
        self.assertIn('TIMEOUT', sim.params)

    def test_params_copy(self):
        u"""params retourne une copie (pas de mutation externe)."""
        sim = Simulation(self.profil)
        p1 = sim.params
        p1['RE'] = 999
        self.assertNotEqual(sim.params['RE'], 999)

    def test_work_dir_auto(self):
        u"""work_dir auto-genere si non specifie."""
        sim = Simulation(self.profil)
        self.assertTrue(os.path.isdir(sim.work_dir))

    def test_work_dir_custom(self):
        u"""work_dir personnalise."""
        import tempfile
        d = tempfile.mkdtemp(prefix='test_sim_')
        sim = Simulation(self.profil, work_dir=d)
        self.assertEqual(sim.work_dir, d)

    def test_invalid_profil(self):
        u"""TypeError si profil n'est pas un Profil."""
        with self.assertRaises(TypeError):
            Simulation("pas un profil")

    def test_reset(self):
        u"""reset() remet a l'etat idle."""
        sim = Simulation(self.profil)
        # Simuler un etat done manuellement
        sim._state = 'done'
        sim._results = SimulationResults()
        sim._error = 'test'
        sim.reset()
        self.assertEqual(sim.state, 'idle')
        self.assertIsNone(sim.results)
        self.assertIsNone(sim.error)

    def test_reset_returns_self(self):
        u"""reset() retourne self (chainage)."""
        sim = Simulation(self.profil)
        self.assertIs(sim.reset(), sim)

    def test_repr(self):
        u"""__repr__ lisible."""
        sim = Simulation(self.profil)
        r = repr(sim)
        self.assertIn('NACA 2412', r)
        self.assertIn('xfoil', r)
        self.assertIn('idle', r)


# ======================================================================
#  Tests plots (non-bloquants, backend Agg)
# ======================================================================

class TestSimulationPlots(unittest.TestCase):
    u"""Tests des methodes de trace."""

    @classmethod
    def setUpClass(cls):
        u"""Forcer le backend non-interactif."""
        import matplotlib
        matplotlib.use('Agg')

    def setUp(self):
        raw = make_fake_results([500000.0, 1000000.0])
        self.results = SimulationResults.from_dict(raw)

    def test_plot_polars(self):
        u"""plot_polars retourne une Figure."""
        from matplotlib.figure import Figure
        fig = self.results.plot_polars(show=False)
        self.assertIsInstance(fig, Figure)
        # 4 subplots
        self.assertEqual(len(fig.axes), 4)

    def test_plot_cl(self):
        u"""plot_cl retourne un Axes."""
        import matplotlib
        ax = self.results.plot_cl(show=False)
        self.assertIsNotNone(ax)

    def test_plot_cd(self):
        u"""plot_cd retourne un Axes."""
        ax = self.results.plot_cd(show=False)
        self.assertIsNotNone(ax)

    def test_plot_finesse(self):
        u"""plot_finesse retourne un Axes."""
        ax = self.results.plot_finesse(show=False)
        self.assertIsNotNone(ax)

    def test_plot_drag_polar(self):
        u"""plot_drag_polar retourne un Axes."""
        ax = self.results.plot_drag_polar(show=False)
        self.assertIsNotNone(ax)

    def test_plot_cp(self):
        u"""plot_cp retourne un Axes."""
        ax = self.results.plot_cp(5.0, show=False)
        self.assertIsNotNone(ax)

    def test_compare_cl(self):
        u"""Superposer 2 SimulationResults sur le meme Axes."""
        raw2 = make_fake_results([500000.0])
        results2 = SimulationResults.from_dict(raw2)

        ax = self.results.plot_cl(label='Sim1', show=False)
        ax = results2.plot_cl(ax=ax, label='Sim2', show=False)

        # Verifier qu'il y a au moins 3 courbes (2 Re de sim1 + 1 Re de sim2)
        lines = ax.get_lines()
        self.assertGreaterEqual(len(lines), 3)

    def test_plot_cl_single_re(self):
        u"""plot_cl avec un Re specifique."""
        ax = self.results.plot_cl(re=500000.0, show=False)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 1)


if __name__ == '__main__':
    unittest.main()
