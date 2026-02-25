#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Tests pour la classe ProfilSpline.

Lance :
    python test_profil_spline.py

@author: Nervures
@date: 2026-02
"""

import os
import sys
import tempfile
import unittest
import math

import numpy as np

# Ajouter le repertoire sources/ au path
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_here, '..', 'sources'))
if _src not in sys.path:
    sys.path.insert(0, _src)

from model.profil_spline import ProfilSpline
from model.bezier import Bezier
from model.bezier_spline import BezierSpline


# ======================================================================
#  Tests construction
# ======================================================================

class TestProfilSplineConstruction(unittest.TestCase):
    u"""Tests de construction du ProfilSpline."""

    def test_from_points(self):
        u"""Construction depuis un tableau de points."""
        pts = np.array([[100, 0], [50, 10], [0, 0], [50, -10], [100, 0]])
        p = ProfilSpline(pts, name='Test')
        self.assertEqual(p.name, 'Test')
        self.assertEqual(len(p.points), 5)

    def test_from_naca_4digits(self):
        u"""NACA 2412 : creation OK."""
        p = ProfilSpline.from_naca('2412')
        self.assertIsInstance(p, ProfilSpline)
        self.assertEqual(p.name, 'NACA 2412')
        self.assertTrue(len(p.points) > 50)

    def test_from_naca_5digits(self):
        u"""NACA 23012 : creation OK."""
        p = ProfilSpline.from_naca('23012')
        self.assertIsInstance(p, ProfilSpline)
        self.assertEqual(p.name, 'NACA 23012')

    def test_invalid_points_raises(self):
        u"""Points invalides levent ValueError."""
        with self.assertRaises(ValueError):
            ProfilSpline(np.array([1, 2, 3]))

    def test_repr(self):
        u"""__repr__ contient le nom."""
        p = ProfilSpline.from_naca('0012')
        r = repr(p)
        self.assertIn('NACA 0012', r)
        self.assertIn('ProfilSpline', r)

    def test_invalid_naca_raises(self):
        u"""Designation NACA invalide leve ValueError."""
        with self.assertRaises(ValueError):
            ProfilSpline.from_naca('123')


# ======================================================================
#  Tests geometrie
# ======================================================================

class TestProfilSplineGeometry(unittest.TestCase):
    u"""Tests des proprietes geometriques."""

    def setUp(self):
        self.p = ProfilSpline.from_naca('2412')

    def test_leading_edge(self):
        u"""Bord d'attaque existe."""
        le = self.p.leading_edge
        self.assertEqual(le.shape, (2,))

    def test_trailing_edge(self):
        u"""Bord de fuite existe."""
        te = self.p.trailing_edge
        self.assertEqual(te.shape, (2,))

    def test_chord(self):
        u"""Corde ~1000 mm pour un NACA."""
        self.assertAlmostEqual(self.p.chord, 1000.0, delta=5.0)

    def test_calage(self):
        u"""Calage ~0 pour un NACA genere."""
        self.assertAlmostEqual(self.p.calage, 0.0, delta=1.0)

    def test_thickness_2412(self):
        u"""Epaisseur relative ~12%."""
        self.assertAlmostEqual(
            self.p.relative_thickness, 0.12, delta=0.02)

    def test_camber_2412(self):
        u"""Cambrure relative ~2%."""
        self.assertAlmostEqual(
            self.p.relative_camber, 0.02, delta=0.01)

    def test_symmetric_camber(self):
        u"""NACA 0012 : cambrure ~0."""
        p = ProfilSpline.from_naca('0012')
        self.assertAlmostEqual(p.relative_camber, 0.0, places=3)

    def test_extrados_intrados(self):
        u"""Extrados et intrados sont des arrays (n, 2)."""
        ext = self.p.extrados
        intr = self.p.intrados
        self.assertEqual(ext.ndim, 2)
        self.assertEqual(ext.shape[1], 2)
        self.assertEqual(intr.ndim, 2)
        self.assertEqual(intr.shape[1], 2)

    def test_extrados_x_increasing(self):
        u"""Extrados : x croissant (BA -> BF)."""
        ext = self.p.extrados
        dx = np.diff(ext[:, 0])
        self.assertTrue(np.all(dx >= -1e-6))

    def test_intrados_x_increasing(self):
        u"""Intrados : x croissant (BA -> BF)."""
        intr = self.p.intrados
        dx = np.diff(intr[:, 0])
        self.assertTrue(np.all(dx >= -1e-6))


# ======================================================================
#  Tests normalisation
# ======================================================================

class TestProfilSplineNormalize(unittest.TestCase):
    u"""Tests de normalisation."""

    def test_already_normalized(self):
        u"""Un NACA genere est deja normalise."""
        p = ProfilSpline.from_naca('0012')
        self.assertTrue(p.is_normalized)

    def test_normalize_after_scale(self):
        u"""Normalisation apres mise a l'echelle."""
        p = ProfilSpline.from_naca('0012')
        p.scale(2.0)
        self.assertFalse(p.is_normalized)
        p.normalize()
        self.assertTrue(p.is_normalized)

    def test_normalize_after_rotate(self):
        u"""Normalisation apres rotation."""
        p = ProfilSpline.from_naca('2412')
        p.rotate(10.0)
        p.normalize()
        self.assertTrue(p.is_normalized)

    def test_normalize_returns_self(self):
        u"""normalize() retourne self."""
        p = ProfilSpline.from_naca('0012')
        self.assertIs(p.normalize(), p)


# ======================================================================
#  Tests transformations
# ======================================================================

class TestProfilSplineTransformations(unittest.TestCase):
    u"""Tests des transformations."""

    def test_translate(self):
        u"""Translation deplace les points."""
        p = ProfilSpline.from_naca('0012')
        le_before = p.leading_edge.copy()
        p.translate(100, 50)
        np.testing.assert_allclose(
            p.leading_edge, le_before + [100, 50], atol=1.0)

    def test_translate_returns_self(self):
        u"""translate() retourne self."""
        p = ProfilSpline.from_naca('0012')
        self.assertIs(p.translate(1, 1), p)

    def test_scale(self):
        u"""scale(2) double la corde."""
        p = ProfilSpline.from_naca('0012')
        chord_before = p.chord
        p.scale(2.0)
        self.assertAlmostEqual(p.chord, chord_before * 2.0, delta=1.0)

    def test_scale_returns_self(self):
        u"""scale() retourne self."""
        p = ProfilSpline.from_naca('0012')
        self.assertIs(p.scale(1.5), p)

    def test_rotate(self):
        u"""Rotation change le calage."""
        p = ProfilSpline.from_naca('0012')
        p.rotate(10.0)
        self.assertAlmostEqual(p.calage, 10.0, delta=1.0)

    def test_rotate_returns_self(self):
        u"""rotate() retourne self."""
        p = ProfilSpline.from_naca('0012')
        self.assertIs(p.rotate(5), p)

    def test_chaining(self):
        u"""Transformations chainables."""
        p = ProfilSpline.from_naca('0012')
        result = p.translate(10, 0).scale(2.0).rotate(5)
        self.assertIs(result, p)


# ======================================================================
#  Tests mode spline
# ======================================================================

class TestProfilSplineSpline(unittest.TestCase):
    u"""Tests du mode spline (BezierSpline)."""

    def setUp(self):
        self.p = ProfilSpline.from_naca('0012')

    def test_initial_no_splines(self):
        u"""Pas de splines au depart."""
        self.assertFalse(self.p.has_splines)
        self.assertIsNone(self.p.spline_extrados)
        self.assertIsNone(self.p.spline_intrados)

    def test_approximate_spline(self):
        u"""approximate_spline cree les splines."""
        self.p.approximate_spline(degree=6)
        self.assertTrue(self.p.has_splines)
        self.assertIsInstance(
            self.p.spline_extrados, BezierSpline)
        self.assertIsInstance(
            self.p.spline_intrados, BezierSpline)

    def test_approximate_spline_returns_self(self):
        u"""approximate_spline retourne self."""
        self.assertIs(self.p.approximate_spline(degree=6), self.p)

    def test_spline_single_segment(self):
        u"""approximate_spline cree des splines a 1 segment."""
        self.p.approximate_spline(degree=6)
        self.assertEqual(self.p.spline_extrados.n_segments, 1)
        self.assertEqual(self.p.spline_intrados.n_segments, 1)

    def test_spline_degree(self):
        u"""Le degre du segment est correct."""
        self.p.approximate_spline(degree=8)
        self.assertEqual(self.p.spline_extrados.degree, 8)
        self.assertEqual(self.p.spline_intrados.degree, 8)

    def test_points_in_spline_mode(self):
        u"""points est reconstruit depuis les splines."""
        pts_before = self.p.points.copy()
        self.p.approximate_spline(degree=8, max_dev=0.5)
        pts_spline = self.p.points
        # La forme est correcte
        self.assertEqual(pts_spline.ndim, 2)
        self.assertEqual(pts_spline.shape[1], 2)
        # Le premier et dernier point sont proches du BF
        np.testing.assert_allclose(
            pts_spline[0], pts_spline[-1], atol=5.0)

    def test_extrados_from_spline(self):
        u"""extrados vient de la spline en mode spline."""
        self.p.approximate_spline(degree=6)
        ext = self.p.extrados
        ext_spline = self.p.spline_extrados.points
        np.testing.assert_array_equal(ext, ext_spline)

    def test_intrados_from_spline(self):
        u"""intrados vient de la spline en mode spline."""
        self.p.approximate_spline(degree=6)
        intr = self.p.intrados
        intr_spline = self.p.spline_intrados.points
        np.testing.assert_array_equal(intr, intr_spline)

    def test_clear_splines(self):
        u"""clear_splines revient au mode discret."""
        self.p.approximate_spline(degree=6)
        self.assertTrue(self.p.has_splines)
        self.p.clear_splines()
        self.assertFalse(self.p.has_splines)

    def test_clear_splines_returns_self(self):
        u"""clear_splines retourne self."""
        self.assertIs(self.p.clear_splines(), self.p)

    def test_set_points_clears_splines(self):
        u"""Setter points efface les splines."""
        self.p.approximate_spline(degree=6)
        self.assertTrue(self.p.has_splines)
        self.p.points = self.p.points
        self.assertFalse(self.p.has_splines)

    def test_approximate_find_degree(self):
        u"""degree='find' avec max_dev."""
        self.p.approximate_spline(degree='find', max_dev=1.0)
        self.assertTrue(self.p.has_splines)

    def test_ba_vertical_tangent(self):
        u"""Tangente verticale au BA."""
        self.p.approximate_spline(degree=8)
        # Le premier segment de l'extrados
        seg_ext = self.p.spline_extrados.segments[0]
        tg = seg_ext.tangent(0.0)
        # Tangente quasi-verticale : composante x ~0
        self.assertAlmostEqual(abs(tg[0]), 0.0, delta=0.1)


# ======================================================================
#  Tests deviation
# ======================================================================

class TestProfilSplineDeviation(unittest.TestCase):
    u"""Tests de la methode deviation."""

    def test_deviation_self(self):
        u"""Deviation d'un profil avec lui-meme ~0."""
        p = ProfilSpline.from_naca('0012')
        d = ProfilSpline.deviation(p, p)
        self.assertIn('x_ext', d)
        self.assertIn('y_current_ext', d)
        self.assertIn('y_reference_ext', d)
        np.testing.assert_allclose(
            d['y_current_ext'], d['y_reference_ext'], atol=1e-10)
        np.testing.assert_allclose(
            d['y_current_int'], d['y_reference_int'], atol=1e-10)


# ======================================================================
#  Tests I/O selig, lednicer, csv
# ======================================================================

class TestProfilSplineIO(unittest.TestCase):
    u"""Tests d'ecriture/lecture standard."""

    def setUp(self):
        self.p = ProfilSpline.from_naca('0012')
        self.tmpdir = tempfile.mkdtemp()

    def test_write_read_selig(self):
        u"""Ecriture puis relecture Selig."""
        path = os.path.join(self.tmpdir, 'test.dat')
        self.p.write(path, fmt='selig')
        p2 = ProfilSpline.from_file(path, fmt='selig')
        self.assertAlmostEqual(p2.chord, self.p.chord, delta=1.0)

    def test_write_read_lednicer(self):
        u"""Ecriture puis relecture Lednicer."""
        path = os.path.join(self.tmpdir, 'test_led.dat')
        self.p.write(path, fmt='lednicer')
        p2 = ProfilSpline.from_file(path, fmt='lednicer')
        self.assertAlmostEqual(p2.chord, self.p.chord, delta=5.0)

    def test_write_read_csv(self):
        u"""Ecriture puis relecture CSV."""
        path = os.path.join(self.tmpdir, 'test.csv')
        self.p.write(path, fmt='csv')
        p2 = ProfilSpline.from_file(path, fmt='csv')
        self.assertAlmostEqual(p2.chord, self.p.chord, delta=5.0)

    def test_write_no_path_raises(self):
        u"""Ecriture sans chemin leve ValueError."""
        with self.assertRaises(ValueError):
            self.p.write()

    def test_write_unknown_format_raises(self):
        u"""Format inconnu leve ValueError."""
        with self.assertRaises(ValueError):
            self.p.write('/tmp/x.dat', fmt='xyz')

    def test_from_file_missing_raises(self):
        u"""Fichier inexistant leve IOError."""
        with self.assertRaises(IOError):
            ProfilSpline.from_file('/nonexistent/file.dat')

    def test_output_format_setter(self):
        u"""output_format accepte bspl."""
        self.p.output_format = 'bspl'
        self.assertEqual(self.p.output_format, 'bspl')

    def test_output_format_invalid_raises(self):
        u"""output_format invalide leve ValueError."""
        with self.assertRaises(ValueError):
            self.p.output_format = 'xyz'

    def test_auto_detect_selig(self):
        u"""Detection automatique du format Selig."""
        path = os.path.join(self.tmpdir, 'test_auto.dat')
        self.p.write(path, fmt='selig')
        p2 = ProfilSpline.from_file(path)
        self.assertAlmostEqual(p2.chord, self.p.chord, delta=1.0)


# ======================================================================
#  Tests I/O bspl (mono et multi-segment)
# ======================================================================

class TestProfilSplineBspl(unittest.TestCase):
    u"""Tests du format .bspl."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_write_read_bspl_single_segment(self):
        u"""Ecriture puis relecture .bspl mono-segment."""
        p = ProfilSpline.from_naca('0012')
        p.approximate_spline(degree=8, max_dev=0.5)

        path = os.path.join(self.tmpdir, 'test.bspl')
        p.write(path, fmt='bspl')

        p2 = ProfilSpline.from_file(path)
        self.assertTrue(p2.has_splines)
        self.assertEqual(p2.spline_extrados.n_segments, 1)
        self.assertEqual(p2.spline_intrados.n_segments, 1)
        self.assertAlmostEqual(p2.chord, p.chord, delta=5.0)

    def test_write_bspl_no_splines_raises(self):
        u"""Ecriture bspl sans splines leve ValueError."""
        p = ProfilSpline.from_naca('0012')
        path = os.path.join(self.tmpdir, 'test.bspl')
        with self.assertRaises(ValueError):
            p.write(path, fmt='bspl')

    def test_write_read_bspl_multi_segment(self):
        u"""Ecriture puis relecture .bspl multi-segment."""
        p = ProfilSpline.from_naca('0012')
        p.approximate_spline(degree=8)

        # Obtenir le Bezier de l'extrados, le splitter, creer une
        # BezierSpline a 2 segments
        seg_ext = p.spline_extrados.segments[0]
        spl2 = seg_ext.split(0.5)
        p._spline_extrados = spl2

        seg_int = p.spline_intrados.segments[0]
        spl2_int = seg_int.split(0.5)
        p._spline_intrados = spl2_int

        self.assertEqual(p.spline_extrados.n_segments, 2)
        self.assertEqual(p.spline_intrados.n_segments, 2)

        path = os.path.join(self.tmpdir, 'test_multi.bspl')
        p.write(path, fmt='bspl')

        p2 = ProfilSpline.from_file(path)
        self.assertTrue(p2.has_splines)
        self.assertEqual(p2.spline_extrados.n_segments, 2)
        self.assertEqual(p2.spline_intrados.n_segments, 2)
        # Continuites preservees
        self.assertEqual(p2.spline_extrados.continuities, ['C2'])
        self.assertEqual(p2.spline_intrados.continuities, ['C2'])

    def test_auto_detect_bspl(self):
        u"""Detection automatique du format .bspl."""
        p = ProfilSpline.from_naca('0012')
        p.approximate_spline(degree=6)
        path = os.path.join(self.tmpdir, 'test.bspl')
        p.write(path, fmt='bspl')

        p2 = ProfilSpline.from_file(path)
        self.assertTrue(p2.has_splines)

    def test_bspl_roundtrip_geometry(self):
        u"""Les points sont proches apres roundtrip bspl."""
        p = ProfilSpline.from_naca('2412')
        p.approximate_spline(degree=8, max_dev=0.5)
        pts_before = p.points.copy()

        path = os.path.join(self.tmpdir, 'rt.bspl')
        p.write(path, fmt='bspl')
        p2 = ProfilSpline.from_file(path)
        pts_after = p2.points

        # Les endpoints doivent etre proches
        np.testing.assert_allclose(
            pts_before[0], pts_after[0], atol=5.0)
        np.testing.assert_allclose(
            pts_before[-1], pts_after[-1], atol=5.0)


# ======================================================================
#  Tests retrocompatibilite .bez
# ======================================================================

class TestProfilSplineBezCompat(unittest.TestCase):
    u"""Tests de retrocompatibilite .bez."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _write_bez_file(self, path):
        u"""Ecrit un fichier .bez avec Profil pour tester la lecture."""
        # Creer manuellement un .bez (format Profil original)
        b_ext = Bezier(
            [[0, 0], [0, 60], [300, 80], [600, 40], [1000, 1]])
        b_int = Bezier(
            [[0, 0], [0, -60], [300, -80], [600, -40], [1000, -1]])
        cp_ext = b_ext.control_points
        cp_int = b_int.control_points

        with open(path, 'w') as f:
            f.write('Test Profile\n')
            f.write('EXTRADOS %d\n' % len(cp_ext))
            for i in range(len(cp_ext)):
                f.write(' %12.8f %12.8f\n'
                        % (cp_ext[i, 0], cp_ext[i, 1]))
            f.write('INTRADOS %d\n' % len(cp_int))
            for i in range(len(cp_int)):
                f.write(' %12.8f %12.8f\n'
                        % (cp_int[i, 0], cp_int[i, 1]))

    def test_read_bez_creates_splines(self):
        u"""Lecture .bez cree des BezierSpline mono-segment."""
        path = os.path.join(self.tmpdir, 'test.bez')
        self._write_bez_file(path)

        p = ProfilSpline.from_file(path)
        self.assertTrue(p.has_splines)
        self.assertIsInstance(p.spline_extrados, BezierSpline)
        self.assertIsInstance(p.spline_intrados, BezierSpline)
        self.assertEqual(p.spline_extrados.n_segments, 1)
        self.assertEqual(p.spline_intrados.n_segments, 1)

    def test_read_bez_name(self):
        u"""Le nom est lu correctement depuis .bez."""
        path = os.path.join(self.tmpdir, 'test.bez')
        self._write_bez_file(path)

        p = ProfilSpline.from_file(path)
        self.assertEqual(p.name, 'Test Profile')

    def test_read_bez_auto_detect(self):
        u"""Detection automatique du .bez."""
        path = os.path.join(self.tmpdir, 'test.bez')
        self._write_bez_file(path)

        p = ProfilSpline.from_file(path)
        self.assertTrue(p.has_splines)


# ======================================================================
#  Tests properties additionnelles
# ======================================================================

class TestProfilSplineProperties(unittest.TestCase):
    u"""Tests des properties de base."""

    def test_name_setter(self):
        u"""name est modifiable."""
        p = ProfilSpline.from_naca('0012')
        p.name = 'Nouveau'
        self.assertEqual(p.name, 'Nouveau')

    def test_output_path(self):
        u"""output_path est accessible."""
        p = ProfilSpline.from_naca('0012')
        self.assertIsNone(p.output_path)
        p.output_path = '/tmp/test.dat'
        self.assertEqual(p.output_path, '/tmp/test.dat')

    def test_is_normalized_true(self):
        u"""NACA genere est normalise."""
        p = ProfilSpline.from_naca('0012')
        self.assertTrue(p.is_normalized)


if __name__ == '__main__':
    unittest.main()
