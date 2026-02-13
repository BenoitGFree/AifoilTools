#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Tests pour la classe Profil.

Lance :
    python test_profil.py

@author: Nervures
@date: 2026-02
"""

import os
import sys
import tempfile
import unittest
import math

import numpy as np

# Ajouter le repertoire airfoiltools/ au path
_here = os.path.dirname(os.path.abspath(__file__))
_pkg = os.path.normpath(os.path.join(_here, '..', 'airfoiltools'))
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)

from profil import Profil


class TestProfilNaca(unittest.TestCase):
    u"""Tests de generation NACA."""

    def test_naca_4digits_creation(self):
        u"""NACA 2412 : creation OK."""
        p = Profil.from_naca('2412')
        self.assertIsInstance(p, Profil)
        self.assertEqual(p.name, 'NACA 2412')
        self.assertTrue(len(p.points) > 50)

    def test_naca_0012_symmetric(self):
        u"""NACA 0012 : profil symetrique, cambrure ~0."""
        p = Profil.from_naca('0012')
        self.assertAlmostEqual(p.relative_camber, 0.0, places=3)
        self.assertAlmostEqual(p.relative_thickness, 0.12, delta=0.01)

    def test_naca_2412_properties(self):
        u"""NACA 2412 : epaisseur ~12%, cambrure ~2%."""
        p = Profil.from_naca('2412')
        self.assertAlmostEqual(p.relative_thickness, 0.12, delta=0.02)
        self.assertAlmostEqual(p.relative_camber, 0.02, delta=0.01)

    def test_naca_5digits_creation(self):
        u"""NACA 23012 : creation OK."""
        p = Profil.from_naca('23012')
        self.assertIsInstance(p, Profil)
        self.assertEqual(p.name, 'NACA 23012')

    def test_naca_chord_1000mm(self):
        u"""NACA genere avec corde = 1000 mm."""
        p = Profil.from_naca('2412')
        self.assertAlmostEqual(p.chord, 1000.0, delta=5.0)

    def test_naca_invalid(self):
        u"""Designation NACA invalide -> ValueError."""
        with self.assertRaises(ValueError):
            Profil.from_naca('12')


class TestProfilGeometry(unittest.TestCase):
    u"""Tests des proprietes geometriques."""

    def setUp(self):
        self.profil = Profil.from_naca('2412')

    def test_leading_edge(self):
        u"""BA : x proche de 0."""
        le = self.profil.leading_edge
        self.assertEqual(le.shape, (2,))
        self.assertAlmostEqual(le[0], 0.0, delta=5.0)

    def test_trailing_edge(self):
        u"""BF : x proche de 1000."""
        te = self.profil.trailing_edge
        self.assertEqual(te.shape, (2,))
        self.assertAlmostEqual(te[0], 1000.0, delta=5.0)

    def test_chord_positive(self):
        u"""Corde > 0."""
        self.assertGreater(self.profil.chord, 0)

    def test_calage_small(self):
        u"""Calage proche de 0 pour un NACA genere."""
        self.assertAlmostEqual(self.profil.calage, 0.0, delta=2.0)

    def test_is_normalized(self):
        u"""NACA genere est quasi normalise."""
        p = Profil.from_naca('0012')
        p.normalize()
        self.assertTrue(p.is_normalized)


class TestProfilTransformations(unittest.TestCase):
    u"""Tests des transformations geometriques."""

    def setUp(self):
        self.profil = Profil.from_naca('0012')
        self.profil.normalize()

    def test_scale(self):
        u"""Scale x2 : corde double."""
        c_orig = self.profil.chord
        self.profil.scale(2.0)
        self.assertAlmostEqual(self.profil.chord, 2.0 * c_orig, delta=1.0)

    def test_scale_returns_self(self):
        u"""Scale retourne self (chainage)."""
        result = self.profil.scale(1.5)
        self.assertIs(result, self.profil)

    def test_translate(self):
        u"""Translation de (100, 50)."""
        le_before = self.profil.leading_edge.copy()
        self.profil.translate(100, 50)
        le_after = self.profil.leading_edge
        self.assertAlmostEqual(le_after[0], le_before[0] + 100, delta=0.1)
        self.assertAlmostEqual(le_after[1], le_before[1] + 50, delta=0.1)

    def test_rotate(self):
        u"""Rotation de 10 deg : calage change."""
        self.profil.rotate(10.0)
        self.assertAlmostEqual(self.profil.calage, 10.0, delta=1.0)

    def test_normalize(self):
        u"""Apres scale+translate+rotate, normalize ramene a l'etat standard."""
        self.profil.scale(3.0).translate(200, -100).rotate(15.0)
        self.profil.normalize()
        self.assertTrue(self.profil.is_normalized)

    def test_chaining(self):
        u"""Chainage de transformations."""
        result = self.profil.translate(10, 20).scale(0.5).rotate(5)
        self.assertIs(result, self.profil)


class TestProfilIO(unittest.TestCase):
    u"""Tests lecture/ecriture."""

    def test_write_selig_read_back(self):
        u"""Ecriture Selig puis relecture : points identiques."""
        p1 = Profil.from_naca('2412')
        tmp = os.path.join(tempfile.mkdtemp(), 'test.dat')
        p1.write(tmp, fmt='selig')

        p2 = Profil.from_file(tmp, fmt='selig', unit='mm')
        np.testing.assert_allclose(p1.points, p2.points, atol=1e-4)
        self.assertEqual(p2.name, p1.name)

    def test_write_lednicer_read_back(self):
        u"""Ecriture Lednicer puis relecture."""
        p1 = Profil.from_naca('0012')
        tmp = os.path.join(tempfile.mkdtemp(), 'test.dat')
        p1.write(tmp, fmt='lednicer')

        p2 = Profil.from_file(tmp, fmt='lednicer', unit='mm')
        # Nombre de points peut differer d'1 (jonction BA)
        self.assertAlmostEqual(p1.chord, p2.chord, delta=1.0)

    def test_write_csv_read_back(self):
        u"""Ecriture CSV puis relecture."""
        p1 = Profil.from_naca('2412')
        tmp = os.path.join(tempfile.mkdtemp(), 'test.csv')
        p1.write(tmp, fmt='csv')

        p2 = Profil.from_file(tmp, fmt='csv', unit='mm')
        np.testing.assert_allclose(p1.points, p2.points, atol=1e-4)

    def test_auto_format_detection(self):
        u"""Detection automatique du format Selig."""
        p1 = Profil.from_naca('2412')
        tmp = os.path.join(tempfile.mkdtemp(), 'test.dat')
        p1.write(tmp, fmt='selig')

        p2 = Profil.from_file(tmp, fmt='auto', unit='mm')
        self.assertEqual(p2.output_format, 'selig')

    def test_write_no_path_raises(self):
        u"""Ecriture sans chemin -> ValueError."""
        p = Profil.from_naca('2412')
        with self.assertRaises(ValueError):
            p.write()

    def test_repr(self):
        u"""__repr__ lisible."""
        p = Profil.from_naca('2412')
        r = repr(p)
        self.assertIn('NACA 2412', r)
        self.assertIn('pts', r)


class TestProfilDirect(unittest.TestCase):
    u"""Tests de creation directe."""

    def test_from_array(self):
        u"""Creation depuis un ndarray."""
        pts = np.array([[0, 0], [100, 10], [200, 0], [100, -10]])
        p = Profil(pts, name='Test')
        self.assertEqual(len(p.points), 4)

    def test_invalid_shape_raises(self):
        u"""Shape invalide -> ValueError."""
        with self.assertRaises(ValueError):
            Profil(np.array([1, 2, 3]))

    def test_output_format_setter(self):
        u"""Setter output_format valide/invalide."""
        p = Profil.from_naca('0012')
        p.output_format = 'csv'
        self.assertEqual(p.output_format, 'csv')
        with self.assertRaises(ValueError):
            p.output_format = 'xyz'


class TestProfilBezier(unittest.TestCase):
    u"""Tests du mode Bezier."""

    def setUp(self):
        self.profil = Profil.from_naca('2412')
        self.profil.normalize()

    def test_has_beziers_default_false(self):
        u"""Par defaut, pas de Beziers."""
        self.assertFalse(self.profil.has_beziers)
        self.assertIsNone(self.profil.bezier_extrados)
        self.assertIsNone(self.profil.bezier_intrados)

    def test_approximate_bezier(self):
        u"""approximate_bezier active le mode Bezier."""
        self.profil.approximate_bezier(degree=8)
        self.assertTrue(self.profil.has_beziers)
        self.assertIsNotNone(self.profil.bezier_extrados)
        self.assertIsNotNone(self.profil.bezier_intrados)

    def test_approximate_bezier_returns_self(self):
        u"""approximate_bezier retourne self (chainage)."""
        result = self.profil.approximate_bezier(degree=5)
        self.assertIs(result, self.profil)

    def test_bezier_orientation_ba_to_bf(self):
        u"""Les Beziers vont du BA (x~0) au BF (x~1000)."""
        self.profil.approximate_bezier(degree=8)
        b_ext = self.profil.bezier_extrados
        b_int = self.profil.bezier_intrados
        # P0 ~ BA (x~0)
        self.assertAlmostEqual(b_ext.start_cpoint[0], 0.0, delta=5.0)
        self.assertAlmostEqual(b_int.start_cpoint[0], 0.0, delta=5.0)
        # Pn ~ BF (x~1000)
        self.assertAlmostEqual(b_ext.end_cpoint[0], 1000.0, delta=5.0)
        self.assertAlmostEqual(b_int.end_cpoint[0], 1000.0, delta=5.0)

    def test_extrados_intrados_from_beziers(self):
        u"""En mode Bezier, extrados/intrados retournent les points echantillonnes."""
        ext_discret = self.profil.extrados.copy()
        self.profil.approximate_bezier(degree=10)
        ext_bezier = self.profil.extrados
        int_bezier = self.profil.intrados
        # Les points viennent des Beziers (forme ndarray(n, 2))
        self.assertEqual(ext_bezier.ndim, 2)
        self.assertEqual(ext_bezier.shape[1], 2)
        self.assertEqual(int_bezier.ndim, 2)
        self.assertEqual(int_bezier.shape[1], 2)
        # L'approximation est proche des points discrets
        # (extrados discret a ~100 pts, Bezier aussi par defaut)
        self.assertAlmostEqual(ext_bezier[0, 0], ext_discret[0, 0], delta=1.0)

    def test_points_from_beziers_selig(self):
        u"""En mode Bezier, points reconstruit en convention Selig."""
        self.profil.approximate_bezier(degree=8)
        pts = self.profil.points
        # Convention Selig : BF -> ext -> BA -> int -> BF
        # Premier point ~ BF (x~1000)
        self.assertAlmostEqual(pts[0, 0], 1000.0, delta=5.0)
        # Dernier point ~ BF (x~1000)
        self.assertAlmostEqual(pts[-1, 0], 1000.0, delta=5.0)

    def test_clear_beziers(self):
        u"""clear_beziers revient au mode discret."""
        pts_discret = self.profil.points.copy()
        self.profil.approximate_bezier(degree=8)
        self.assertTrue(self.profil.has_beziers)
        self.profil.clear_beziers()
        self.assertFalse(self.profil.has_beziers)
        # points reviennent aux discrets
        np.testing.assert_array_equal(self.profil.points, pts_discret)

    def test_clear_beziers_returns_self(self):
        u"""clear_beziers retourne self (chainage)."""
        result = self.profil.clear_beziers()
        self.assertIs(result, self.profil)

    def test_points_setter_clears_beziers(self):
        u"""Affecter points supprime les Beziers."""
        self.profil.approximate_bezier(degree=5)
        self.assertTrue(self.profil.has_beziers)
        # Affecter de nouveaux points
        new_pts = self.profil._points.copy()
        self.profil.points = new_pts
        self.assertFalse(self.profil.has_beziers)

    def test_approximate_bezier_find_degree(self):
        u"""degree='find' avec max_dev trouve un degre automatiquement."""
        self.profil.approximate_bezier(degree='find', max_dev=1.0)
        self.assertTrue(self.profil.has_beziers)
        # Les Beziers ont un degre raisonnable (< 30)
        self.assertLess(self.profil.bezier_extrados.degree, 30)
        self.assertLess(self.profil.bezier_intrados.degree, 30)

    def test_approximate_bezier_n_points(self):
        u"""n_points controle l'echantillonnage des Beziers."""
        self.profil.approximate_bezier(degree=8, n_points=50)
        self.assertEqual(len(self.profil.extrados), 50)
        self.assertEqual(len(self.profil.intrados), 50)


if __name__ == '__main__':
    unittest.main()
