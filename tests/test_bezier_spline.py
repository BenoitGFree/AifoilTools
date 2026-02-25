#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Tests pour la classe BezierSpline.

@author: Nervures
@date: 2026-02
"""

import os
import sys
import unittest

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_here, '..', 'sources'))
if _src not in sys.path:
    sys.path.insert(0, _src)

from model.bezier import Bezier
from model.bezier_spline import BezierSpline


# ======================================================================
#  Helpers
# ======================================================================

def _make_two_cubics():
    u"""Cree 2 cubiques C0 connectees en (300, 0)."""
    b1 = Bezier([[0, 0], [100, 200], [200, 200], [300, 0]])
    b2 = Bezier([[300, 0], [400, -100], [500, -100], [600, 0]])
    return b1, b2


# ======================================================================
#  Tests construction
# ======================================================================

class TestBezierSplineConstruction(unittest.TestCase):
    u"""Tests de construction du BezierSpline."""

    def test_single_segment(self):
        u"""Un seul segment, pas de continuite."""
        b = Bezier([[0, 0], [100, 200], [200, 0]])
        sp = BezierSpline([b])
        self.assertEqual(sp.n_segments, 1)
        self.assertEqual(sp.continuities, [])

    def test_two_segments_c0(self):
        u"""Deux segments avec continuite C0."""
        b1, b2 = _make_two_cubics()
        sp = BezierSpline([b1, b2], continuities=['C0'])
        self.assertEqual(sp.n_segments, 2)
        self.assertEqual(sp.continuities, ['C0'])

    def test_default_continuity(self):
        u"""Sans argument, la continuite est C0 par defaut."""
        b1, b2 = _make_two_cubics()
        sp = BezierSpline([b1, b2])
        self.assertEqual(sp.continuities, ['C0'])

    def test_invalid_continuity_raises(self):
        u"""Continuite invalide leve ValueError."""
        b1, b2 = _make_two_cubics()
        with self.assertRaises(ValueError):
            BezierSpline([b1, b2], continuities=['C3'])

    def test_wrong_length_raises(self):
        u"""Mauvaise longueur de continuites leve ValueError."""
        b1, b2 = _make_two_cubics()
        with self.assertRaises(ValueError):
            BezierSpline([b1, b2], continuities=['C0', 'C1'])

    def test_empty_segments_raises(self):
        u"""Liste vide leve ValueError."""
        with self.assertRaises(ValueError):
            BezierSpline([])

    def test_non_bezier_raises(self):
        u"""Element non-Bezier leve TypeError."""
        with self.assertRaises(TypeError):
            BezierSpline(["not a bezier"])


# ======================================================================
#  Tests properties
# ======================================================================

class TestBezierSplineProperties(unittest.TestCase):
    u"""Tests des properties du BezierSpline."""

    def setUp(self):
        b1, b2 = _make_two_cubics()
        self.spline = BezierSpline([b1, b2], continuities=['C0'])

    def test_n_segments(self):
        u"""n_segments retourne 2."""
        self.assertEqual(self.spline.n_segments, 2)

    def test_degree_uniform(self):
        u"""degree retourne un int si tous les segments ont le meme degre."""
        self.assertEqual(self.spline.degree, 3)

    def test_control_points_no_duplicates(self):
        u"""control_points concatene sans doublons aux jonctions."""
        cp = self.spline.control_points
        # 4 + 4 - 1 (jonction) = 7
        self.assertEqual(len(cp), 7)

    def test_start_end_cpoint(self):
        u"""start_cpoint et end_cpoint correspondent."""
        np.testing.assert_allclose(
            self.spline.start_cpoint, [0, 0])
        np.testing.assert_allclose(
            self.spline.end_cpoint, [600, 0])

    def test_name(self):
        u"""Le nom est accessible et modifiable."""
        sp = BezierSpline(
            [Bezier([[0, 0], [1, 1]])], name='Test')
        self.assertEqual(sp.name, 'Test')
        sp.name = 'Nouveau'
        self.assertEqual(sp.name, 'Nouveau')


# ======================================================================
#  Tests evaluation
# ======================================================================

class TestBezierSplineEvaluation(unittest.TestCase):
    u"""Tests d'evaluation du BezierSpline."""

    def setUp(self):
        b1, b2 = _make_two_cubics()
        self.spline = BezierSpline([b1, b2])

    def test_eval_t0(self):
        u"""evaluate(0) retourne le premier point."""
        pt = self.spline.evaluate(0.0)
        np.testing.assert_allclose(pt, [0, 0], atol=1e-10)

    def test_eval_t_N(self):
        u"""evaluate(N) retourne le dernier point."""
        pt = self.spline.evaluate(2.0)
        np.testing.assert_allclose(pt, [600, 0], atol=1e-10)

    def test_eval_junction(self):
        u"""evaluate(1.0) = point de jonction."""
        pt = self.spline.evaluate(1.0)
        np.testing.assert_allclose(pt, [300, 0], atol=1e-10)

    def test_eval_scalar_shape(self):
        u"""Scalaire → ndarray(2,)."""
        pt = self.spline.evaluate(0.5)
        self.assertEqual(pt.shape, (2,))

    def test_eval_vector_shape(self):
        u"""Vecteur → ndarray(m, 2)."""
        pts = self.spline.evaluate([0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(pts.shape, (5, 2))

    def test_eval_out_of_range_raises(self):
        u"""t hors [0, N] leve ValueError."""
        with self.assertRaises(ValueError):
            self.spline.evaluate(-0.1)
        with self.assertRaises(ValueError):
            self.spline.evaluate(2.1)

    def test_tangent_at_junction(self):
        u"""Tangente evaluable a la jonction."""
        tg = self.spline.tangent(1.0)
        self.assertEqual(tg.shape, (2,))
        # Vecteur non nul
        self.assertGreater(np.linalg.norm(tg), 0.5)

    def test_curvature_scalar(self):
        u"""Courbure retourne un scalaire pour t scalaire."""
        k = self.spline.curvature(0.5)
        self.assertIsInstance(float(k), float)

    def test_curvature_vector(self):
        u"""Courbure retourne un array pour t array."""
        k = self.spline.curvature([0.5, 1.0, 1.5])
        self.assertEqual(len(k), 3)


# ======================================================================
#  Tests echantillonnage
# ======================================================================

class TestBezierSplineSampling(unittest.TestCase):
    u"""Tests d'echantillonnage du BezierSpline."""

    def setUp(self):
        b1, b2 = _make_two_cubics()
        self.spline = BezierSpline([b1, b2], n_points=100)

    def test_points_shape(self):
        u"""points retourne n_points lignes."""
        pts = self.spline.points
        self.assertEqual(pts.shape, (100, 2))

    def test_points_endpoints(self):
        u"""Premier et dernier points echantillonnes."""
        pts = self.spline.points
        np.testing.assert_allclose(pts[0], [0, 0], atol=1.0)
        np.testing.assert_allclose(pts[-1], [600, 0], atol=1.0)

    def test_sample_explicit(self):
        u"""sample(50) retourne 50 points."""
        pts = self.spline.sample(50)
        self.assertEqual(len(pts), 50)

    def test_tangents_shape(self):
        u"""tangents retourne n_points lignes."""
        tg = self.spline.tangents
        self.assertEqual(tg.shape, (100, 2))

    def test_normals_shape(self):
        u"""normals retourne n_points lignes."""
        nm = self.spline.normals
        self.assertEqual(nm.shape, (100, 2))

    def test_curvatures_shape(self):
        u"""curvatures retourne n_points valeurs."""
        k = self.spline.curvatures
        self.assertEqual(k.shape, (100,))

    def test_n_points_setter(self):
        u"""Modifier n_points change la taille."""
        self.spline.n_points = 50
        self.assertEqual(len(self.spline.points), 50)


# ======================================================================
#  Tests transformations
# ======================================================================

class TestBezierSplineTransformations(unittest.TestCase):
    u"""Tests des transformations du BezierSpline."""

    def test_translate(self):
        u"""Translation deplace tous les points."""
        b1, b2 = _make_two_cubics()
        sp = BezierSpline([b1, b2])
        start_before = sp.start_cpoint.copy()
        sp.translate(10, 20)
        np.testing.assert_allclose(
            sp.start_cpoint, start_before + [10, 20])

    def test_translate_returns_self(self):
        u"""translate retourne self."""
        b = Bezier([[0, 0], [100, 0]])
        sp = BezierSpline([b])
        self.assertIs(sp.translate(1, 1), sp)

    def test_scale(self):
        u"""Scale x2 double les coordonnees."""
        b = Bezier([[0, 0], [100, 0], [200, 0]])
        sp = BezierSpline([b])
        sp.scale(2.0)
        np.testing.assert_allclose(
            sp.end_cpoint, [400, 0], atol=1e-10)

    def test_reverse_swaps_endpoints(self):
        u"""Reverse echange debut et fin."""
        b1, b2 = _make_two_cubics()
        sp = BezierSpline([b1, b2])
        sp.reverse()
        np.testing.assert_allclose(
            sp.start_cpoint, [600, 0], atol=1e-10)
        np.testing.assert_allclose(
            sp.end_cpoint, [0, 0], atol=1e-10)

    def test_double_reverse_identity(self):
        u"""Deux reverse successifs = identite."""
        b1, b2 = _make_two_cubics()
        sp = BezierSpline([b1, b2])
        cp_before = sp.control_points.copy()
        sp.reverse().reverse()
        np.testing.assert_allclose(
            sp.control_points, cp_before, atol=1e-10)

    def test_chaining(self):
        u"""Les transformations sont chainables."""
        b = Bezier([[0, 0], [100, 0]])
        sp = BezierSpline([b])
        result = sp.translate(10, 0).scale(2.0)
        self.assertIs(result, sp)


# ======================================================================
#  Test repr
# ======================================================================

class TestBezierSplineRepr(unittest.TestCase):
    u"""Test de __repr__."""

    def test_repr(self):
        u"""__repr__ contient le nom et le nombre de segments."""
        b1, b2 = _make_two_cubics()
        sp = BezierSpline([b1, b2], name='Test')
        r = repr(sp)
        self.assertIn('Test', r)
        self.assertIn('2 segments', r)


# ======================================================================
#  Test project
# ======================================================================

class TestBezierSplineProject(unittest.TestCase):
    u"""Tests de la projection sur la spline."""

    def test_project_point_on_curve(self):
        u"""Projeter un point situe sur la courbe : distance ~ 0."""
        b = Bezier([[0, 0], [100, 200], [200, 200], [300, 0]])
        sp = BezierSpline([b])
        # Evaluer la courbe a t=0.5
        pt_on = sp.evaluate(0.5)
        t, pt, dist = sp.project(pt_on)
        self.assertAlmostEqual(dist, 0.0, places=6)
        np.testing.assert_allclose(pt, pt_on, atol=1e-6)
        self.assertAlmostEqual(t, 0.5, places=3)

    def test_project_off_curve(self):
        u"""Projeter un point a cote de la courbe."""
        b = Bezier([[0, 0], [100, 200], [200, 200], [300, 0]])
        sp = BezierSpline([b])
        # Point au-dessus du milieu
        t, pt, dist = sp.project([150, 250])
        self.assertGreater(dist, 0)
        self.assertTrue(0.0 <= t <= 1.0)

    def test_project_two_segments(self):
        u"""Projection sur le 2e segment."""
        b1, b2 = _make_two_cubics()
        sp = BezierSpline([b1, b2])
        # Point proche du milieu du segment 2 (t_global ~ 1.5)
        pt_mid = sp.evaluate(1.5)
        # Decaler legerement
        query = pt_mid + np.array([0, 5.0])
        t, pt, dist = sp.project(query)
        self.assertTrue(1.0 < t < 2.0,
                        "t=%.3f devrait etre dans ]1, 2[" % t)
        self.assertLess(dist, 10.0)

    def test_project_near_start(self):
        u"""Projection pres du debut : t proche de 0."""
        b = Bezier([[0, 0], [100, 200], [200, 200], [300, 0]])
        sp = BezierSpline([b])
        t, pt, dist = sp.project([1, 0])
        self.assertLess(t, 0.1)

    def test_project_near_end(self):
        u"""Projection pres de la fin : t proche de N."""
        b = Bezier([[0, 0], [100, 200], [200, 200], [300, 0]])
        sp = BezierSpline([b])
        t, pt, dist = sp.project([299, 0])
        self.assertGreater(t, 0.9)


# ======================================================================
#  Test split
# ======================================================================

class TestBezierSplineSplit(unittest.TestCase):
    u"""Tests du split de BezierSpline."""

    def test_split_single_returns_two(self):
        u"""Split d'un segment unique donne 2 segments."""
        b = Bezier([[0, 0], [100, 200], [200, 200], [300, 0]])
        sp = BezierSpline([b], n_points=100)
        result = sp.split(0.5)
        self.assertEqual(result.n_segments, 2)
        self.assertEqual(result.continuities, ['C2'])

    def test_split_preserves_geometry(self):
        u"""La geometrie est preservee apres le split."""
        b = Bezier([[0, 0], [100, 200], [200, 200], [300, 0]])
        sp = BezierSpline([b], n_points=200)
        result = sp.split(0.4)
        # Evaluer l'original a N points, projeter sur le split
        t_vals = np.linspace(0, 1, 100)
        pts_orig = sp.evaluate(t_vals)
        for pt in pts_orig:
            _, _, dist = result.project(pt)
            self.assertLess(dist, 1e-6,
                            "Point %.2f, %.2f : dist=%.2e"
                            % (pt[0], pt[1], dist))

    def test_split_continuity_c2(self):
        u"""La nouvelle jonction a continuite C2."""
        b = Bezier([[0, 0], [100, 200], [200, 200], [300, 0]])
        sp = BezierSpline([b])
        result = sp.split(0.5)
        # Tangentes au point de jonction
        seg0 = result.segments[0]
        seg1 = result.segments[1]
        tg_l = seg0.tangent(1.0)
        tg_r = seg1.tangent(0.0)
        cross = abs(tg_l[0] * tg_r[1] - tg_l[1] * tg_r[0])
        self.assertLess(cross, 1e-6)
        # Courbures
        k_l = seg0.curvature(1.0)
        k_r = seg1.curvature(0.0)
        self.assertAlmostEqual(k_l, k_r, places=4)

    def test_split_multi_segment(self):
        u"""Split d'une spline a 3 segments donne 4 segments."""
        b1 = Bezier([[0, 0], [50, 100], [100, 0]])
        b2 = Bezier([[100, 0], [150, -100], [200, 0]])
        b3 = Bezier([[200, 0], [250, 100], [300, 0]])
        sp = BezierSpline([b1, b2, b3],
                          continuities=['C0', 'C1'])
        # Split dans le segment 1 (t_global = 1.5)
        result = sp.split(1.5)
        self.assertEqual(result.n_segments, 4)

    def test_split_preserves_existing_continuities(self):
        u"""Les continuites existantes sont preservees."""
        b1 = Bezier([[0, 0], [50, 100], [100, 0]])
        b2 = Bezier([[100, 0], [150, -100], [200, 0]])
        b3 = Bezier([[200, 0], [250, 100], [300, 0]])
        sp = BezierSpline([b1, b2, b3],
                          continuities=['C0', 'C1'])
        result = sp.split(1.5)
        # Continuites attendues : C0, C2 (nouvelle), C1
        self.assertEqual(result.continuities, ['C0', 'C2', 'C1'])

    def test_split_at_boundary_raises(self):
        u"""Split a t=0 ou t=N leve ValueError."""
        b = Bezier([[0, 0], [100, 200], [200, 0]])
        sp = BezierSpline([b])
        with self.assertRaises(ValueError):
            sp.split(0.0)
        with self.assertRaises(ValueError):
            sp.split(1.0)

    def test_split_at_junction_raises(self):
        u"""Split a une jonction (entier) leve ValueError."""
        b1, b2 = _make_two_cubics()
        sp = BezierSpline([b1, b2])
        with self.assertRaises(ValueError):
            sp.split(1.0)

    def test_split_returns_new_object(self):
        u"""Le split retourne un nouvel objet, l'original est inchange."""
        b = Bezier([[0, 0], [100, 200], [200, 200], [300, 0]])
        sp = BezierSpline([b])
        pts_before = sp.points.copy()
        result = sp.split(0.5)
        self.assertIsNot(result, sp)
        self.assertEqual(sp.n_segments, 1)
        np.testing.assert_array_equal(sp.points, pts_before)

    def test_split_inherits_params(self):
        u"""Le split herite de n_points, sample_mode."""
        b = Bezier([[0, 0], [100, 200], [200, 200], [300, 0]])
        sp = BezierSpline([b], n_points=500,
                          sample_mode='curvilinear')
        result = sp.split(0.5)
        self.assertEqual(result.n_points, 500)
        self.assertEqual(result.sample_mode, 'curvilinear')

    def test_split_out_of_range_raises(self):
        u"""Split hors limites leve ValueError."""
        b = Bezier([[0, 0], [100, 200], [200, 0]])
        sp = BezierSpline([b])
        with self.assertRaises(ValueError):
            sp.split(-0.5)
        with self.assertRaises(ValueError):
            sp.split(1.5)


if __name__ == '__main__':
    unittest.main()
