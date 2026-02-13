#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Tests pour la classe Bezier.

Lance :
    python test_bezier.py

@author: Nervures
@date: 2026-02
"""

import os
import sys
import unittest
import math

import numpy as np

# Ajouter le repertoire sources/ au path
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_here, '..', 'sources'))
if _src not in sys.path:
    sys.path.insert(0, _src)

from model.bezier import Bezier


class TestBezierConstruction(unittest.TestCase):
    u"""Tests de construction."""

    def test_from_list(self):
        u"""Creation depuis une liste de listes."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        self.assertIsInstance(b, Bezier)
        self.assertEqual(b.degree, 3)
        self.assertEqual(len(b.control_points), 4)

    def test_from_ndarray(self):
        u"""Creation depuis un ndarray."""
        pts = np.array([[0, 0], [50, 100], [100, 0]])
        b = Bezier(pts)
        self.assertEqual(b.degree, 2)

    def test_degree_1(self):
        u"""Degre 1 (droite) : 2 points."""
        b = Bezier([[0, 0], [100, 50]])
        self.assertEqual(b.degree, 1)

    def test_invalid_shape_raises(self):
        u"""Shape invalide -> ValueError."""
        with self.assertRaises(ValueError):
            Bezier(np.array([1, 2, 3]))

    def test_too_few_points_raises(self):
        u"""Moins de 2 points -> ValueError."""
        with self.assertRaises(ValueError):
            Bezier([[0, 0]])

    def test_name_default(self):
        u"""Nom par defaut."""
        b = Bezier([[0, 0], [1, 1]])
        self.assertEqual(b.name, 'Sans nom')

    def test_name_custom(self):
        u"""Nom personnalise."""
        b = Bezier([[0, 0], [1, 1]], name='Ma courbe')
        self.assertEqual(b.name, 'Ma courbe')

    def test_repr(self):
        u"""__repr__ lisible."""
        b = Bezier([[0, 0], [1, 2], [3, 4], [5, 0]], name='Test')
        r = repr(b)
        self.assertIn('Test', r)
        self.assertIn('degre=3', r)
        self.assertIn('4 pts', r)


class TestBezierProperties(unittest.TestCase):
    u"""Tests des properties."""

    def setUp(self):
        self.pts = np.array([[0, 0], [100, 200], [300, 200], [400, 0]],
                            dtype=float)
        self.bez = Bezier(self.pts)

    def test_start_cpoint(self):
        u"""start_cpoint = P0."""
        np.testing.assert_array_equal(self.bez.start_cpoint, [0, 0])

    def test_end_cpoint(self):
        u"""end_cpoint = Pn."""
        np.testing.assert_array_equal(self.bez.end_cpoint, [400, 0])

    def test_start_cpoint_is_copy(self):
        u"""start_cpoint retourne une copie."""
        sp = self.bez.start_cpoint
        sp[0] = 999
        np.testing.assert_array_equal(self.bez.start_cpoint, [0, 0])


class TestBezierEvaluation(unittest.TestCase):
    u"""Tests d'evaluation."""

    def test_eval_t0(self):
        u"""B(0) = P0."""
        b = Bezier([[10, 20], [50, 100], [90, 100], [130, 20]])
        pt = b.evaluate(0.0)
        np.testing.assert_allclose(pt, [10, 20], atol=1e-10)

    def test_eval_t1(self):
        u"""B(1) = Pn."""
        b = Bezier([[10, 20], [50, 100], [90, 100], [130, 20]])
        pt = b.evaluate(1.0)
        np.testing.assert_allclose(pt, [130, 20], atol=1e-10)

    def test_eval_line_midpoint(self):
        u"""Droite (degre 1) : B(0.5) = milieu."""
        b = Bezier([[0, 0], [100, 60]])
        pt = b.evaluate(0.5)
        np.testing.assert_allclose(pt, [50, 30], atol=1e-10)

    def test_eval_quadratic_midpoint(self):
        u"""Quadratique symetrique : B(0.5) = point connu."""
        # P0=(0,0), P1=(50,100), P2=(100,0)
        # B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2 = (50, 50)
        b = Bezier([[0, 0], [50, 100], [100, 0]])
        pt = b.evaluate(0.5)
        np.testing.assert_allclose(pt, [50, 50], atol=1e-10)

    def test_eval_vector_t(self):
        u"""Evaluation avec un vecteur de t."""
        b = Bezier([[0, 0], [100, 0]])
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        pts = b.evaluate(t)
        self.assertEqual(pts.shape, (5, 2))
        np.testing.assert_allclose(pts[:, 0], [0, 25, 50, 75, 100], atol=1e-10)
        np.testing.assert_allclose(pts[:, 1], 0, atol=1e-10)

    def test_eval_scalar_returns_1d(self):
        u"""t scalaire -> ndarray(2,)."""
        b = Bezier([[0, 0], [1, 1]])
        pt = b.evaluate(0.5)
        self.assertEqual(pt.shape, (2,))

    def test_eval_symmetric_cubic(self):
        u"""Cubique symetrique : B(0.5) sur l'axe de symetrie."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pt = b.evaluate(0.5)
        self.assertAlmostEqual(pt[0], 200.0, places=5)


class TestBezierDerivatives(unittest.TestCase):
    u"""Tests des derivees."""

    def test_deriv1_line(self):
        u"""Droite (degre 1) : derivee constante = direction."""
        b = Bezier([[0, 0], [100, 60]])
        d = b.derivative(0.0)
        np.testing.assert_allclose(d, [100, 60], atol=1e-10)
        d2 = b.derivative(0.5)
        np.testing.assert_allclose(d2, [100, 60], atol=1e-10)
        d3 = b.derivative(1.0)
        np.testing.assert_allclose(d3, [100, 60], atol=1e-10)

    def test_deriv1_cubic_endpoints(self):
        u"""Cubique : B'(0) = 3*(P1-P0), B'(1) = 3*(P3-P2)."""
        p = np.array([[0, 0], [10, 30], [40, 30], [50, 0]], dtype=float)
        b = Bezier(p)
        d0 = b.derivative(0.0)
        np.testing.assert_allclose(d0, 3 * (p[1] - p[0]), atol=1e-10)
        d1 = b.derivative(1.0)
        np.testing.assert_allclose(d1, 3 * (p[3] - p[2]), atol=1e-10)

    def test_deriv2_line(self):
        u"""Droite : derivee seconde nulle."""
        b = Bezier([[0, 0], [100, 60]])
        d2 = b.derivative(0.5, order=2)
        np.testing.assert_allclose(d2, [0, 0], atol=1e-10)

    def test_deriv2_quadratic(self):
        u"""Quadratique : derivee seconde constante."""
        # B(t) = (1-t)^2*P0 + 2t(1-t)*P1 + t^2*P2
        # B''(t) = 2*(P2 - 2*P1 + P0)
        p = np.array([[0, 0], [50, 100], [100, 0]], dtype=float)
        b = Bezier(p)
        expected = 2 * (p[2] - 2 * p[1] + p[0])  # [0, -400]
        d2_0 = b.derivative(0.0, order=2)
        d2_5 = b.derivative(0.5, order=2)
        np.testing.assert_allclose(d2_0, expected, atol=1e-10)
        np.testing.assert_allclose(d2_5, expected, atol=1e-10)

    def test_deriv_vector_t(self):
        u"""Derivee avec vecteur de t."""
        b = Bezier([[0, 0], [100, 60]])
        d = b.derivative(np.array([0.0, 0.5, 1.0]))
        self.assertEqual(d.shape, (3, 2))

    def test_deriv_invalid_order(self):
        u"""Ordre invalide -> ValueError."""
        b = Bezier([[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            b.derivative(0.5, order=3)


class TestBezierTangentNormal(unittest.TestCase):
    u"""Tests tangente et normale."""

    def test_tangent_unit_norm(self):
        u"""Tangente de norme 1."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        tg = b.tangent(0.5)
        self.assertAlmostEqual(np.linalg.norm(tg), 1.0, places=10)

    def test_tangent_line_horizontal(self):
        u"""Droite horizontale : tangente = (1, 0)."""
        b = Bezier([[0, 0], [100, 0]])
        tg = b.tangent(0.5)
        np.testing.assert_allclose(tg, [1, 0], atol=1e-10)

    def test_normal_orthogonal(self):
        u"""Normale orthogonale a la tangente."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            tg = b.tangent(t)
            nm = b.normal(t)
            dot = np.dot(tg, nm)
            self.assertAlmostEqual(dot, 0.0, places=10)

    def test_normal_unit_norm(self):
        u"""Normale de norme 1."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        nm = b.normal(0.3)
        self.assertAlmostEqual(np.linalg.norm(nm), 1.0, places=10)

    def test_tangent_vector_t(self):
        u"""Tangente avec vecteur de t."""
        b = Bezier([[0, 0], [50, 100], [100, 0]])
        t = np.array([0.0, 0.5, 1.0])
        tgs = b.tangent(t)
        self.assertEqual(tgs.shape, (3, 2))
        norms = np.linalg.norm(tgs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


class TestBezierCurvature(unittest.TestCase):
    u"""Tests de courbure."""

    def test_curvature_line_zero(self):
        u"""Droite : courbure nulle."""
        b = Bezier([[0, 0], [100, 50]])
        k = b.curvature(0.5)
        self.assertAlmostEqual(k, 0.0, places=10)

    def test_curvature_nonzero(self):
        u"""Courbe non triviale : courbure non nulle."""
        b = Bezier([[0, 0], [0, 100], [100, 100], [100, 0]])
        k = b.curvature(0.5)
        self.assertNotAlmostEqual(k, 0.0, places=3)

    def test_curvature_circle_approximation(self):
        u"""Quart de cercle approx. : courbure ~ 1/R."""
        # Approximation cubique d'un quart de cercle de rayon R=100
        # P0=(100,0), P1=(100,55.2), P2=(55.2,100), P3=(0,100)
        k = 0.5522847498  # constante standard
        R = 100.0
        b = Bezier([
            [R, 0],
            [R, k * R],
            [k * R, R],
            [0, R]
        ])
        kappa = b.curvature(0.5)
        # Pour un cercle parfait, kappa = 1/R = 0.01
        self.assertAlmostEqual(abs(kappa), 1.0 / R, delta=0.002)

    def test_curvature_vector_t(self):
        u"""Courbure avec vecteur de t."""
        b = Bezier([[0, 0], [50, 100], [100, 0]])
        t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        k = b.curvature(t)
        self.assertEqual(k.shape, (5,))


class TestBezierHighDegree(unittest.TestCase):
    u"""Tests avec degre eleve."""

    def test_degree_5(self):
        u"""Degre 5 : 6 points de controle."""
        pts = [[0, 0], [20, 50], [40, 80], [60, 80], [80, 50], [100, 0]]
        b = Bezier(pts)
        self.assertEqual(b.degree, 5)
        # B(0) et B(1) corrects
        np.testing.assert_allclose(b.evaluate(0.0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(b.evaluate(1.0), [100, 0], atol=1e-10)

    def test_degree_5_derivatives(self):
        u"""Degre 5 : derivees d'ordre 1 et 2 calculables."""
        pts = [[0, 0], [20, 50], [40, 80], [60, 80], [80, 50], [100, 0]]
        b = Bezier(pts)
        d1 = b.derivative(0.5)
        d2 = b.derivative(0.5, order=2)
        self.assertEqual(d1.shape, (2,))
        self.assertEqual(d2.shape, (2,))


class TestBezierPointsProperty(unittest.TestCase):
    u"""Tests de la property points (echantillonnage auto)."""

    def test_points_default(self):
        u"""points retourne 100 points par defaut."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        self.assertEqual(b.points.shape, (100, 2))

    def test_points_custom_n(self):
        u"""n_points personnalise."""
        b = Bezier([[0, 0], [100, 200], [400, 0]], n_points=50)
        self.assertEqual(b.points.shape, (50, 2))

    def test_points_setter_n(self):
        u"""Modifier n_points change points."""
        b = Bezier([[0, 0], [100, 200], [400, 0]])
        b.n_points = 30
        self.assertEqual(b.points.shape, (30, 2))

    def test_points_adaptive_mode(self):
        u"""Mode adaptive via constructeur."""
        b = Bezier([[0, 0], [0, 100], [100, 100], [100, 0]],
                    n_points=40, sample_mode='adaptive', tolerance=1.0)
        self.assertEqual(b.points.shape, (40, 2))

    def test_points_change_mode(self):
        u"""Changement de mode a la volee."""
        b = Bezier([[0, 0], [0, 100], [100, 100], [100, 0]])
        b.sample_mode = 'adaptive'
        b.tolerance = 1.0
        pts = b.points
        self.assertEqual(pts.shape, (100, 2))

    def test_n_points_invalid_raises(self):
        u"""n_points < 2 -> ValueError."""
        b = Bezier([[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            b.n_points = 1

    def test_sample_mode_invalid_raises(self):
        u"""Mode invalide -> ValueError."""
        b = Bezier([[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            b.sample_mode = 'unknown'

    def test_points_endpoints(self):
        u"""points[0] ~ P0 et points[-1] ~ Pn."""
        b = Bezier([[10, 20], [50, 100], [90, 100], [130, 20]])
        pts = b.points
        np.testing.assert_allclose(pts[0], [10, 20], atol=0.5)
        np.testing.assert_allclose(pts[-1], [130, 20], atol=0.5)


class TestBezierSampleCurvilinear(unittest.TestCase):
    u"""Tests echantillonnage curviligne."""

    def test_sample_shape(self):
        u"""sample(n) retourne ndarray(n, 2)."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = b.sample(50)
        self.assertEqual(pts.shape, (50, 2))

    def test_sample_endpoints(self):
        u"""Premier et dernier points = P0 et Pn."""
        b = Bezier([[10, 20], [50, 100], [90, 100], [130, 20]])
        pts = b.sample(30)
        np.testing.assert_allclose(pts[0], [10, 20], atol=0.5)
        np.testing.assert_allclose(pts[-1], [130, 20], atol=0.5)

    def test_sample_uniform_spacing_line(self):
        u"""Droite : espacement parfaitement uniforme."""
        b = Bezier([[0, 0], [100, 0]])
        pts = b.sample(11)
        # x doit aller de 0 a 100 par pas de 10
        np.testing.assert_allclose(pts[:, 0],
                                   np.linspace(0, 100, 11), atol=0.5)
        np.testing.assert_allclose(pts[:, 1], 0, atol=0.01)

    def test_sample_arc_length_regularity(self):
        u"""Cubique : espacements en abscisse curviligne quasi uniformes."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = b.sample(50)
        ds = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        # Ecart relatif entre le plus grand et le plus petit pas
        ratio = ds.max() / ds.min()
        self.assertLess(ratio, 1.2)  # quasi uniforme

    def test_sample_n2(self):
        u"""n=2 : juste les extremites."""
        b = Bezier([[0, 0], [50, 100], [100, 0]])
        pts = b.sample(2)
        self.assertEqual(pts.shape, (2, 2))
        np.testing.assert_allclose(pts[0], [0, 0], atol=0.5)
        np.testing.assert_allclose(pts[-1], [100, 0], atol=0.5)

    def test_sample_n1_raises(self):
        u"""n < 2 -> ValueError."""
        b = Bezier([[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            b.sample(1)


class TestBezierSampleAdaptive(unittest.TestCase):
    u"""Tests echantillonnage adaptatif."""

    def test_adaptive_shape(self):
        u"""sample adaptive retourne ndarray(n, 2)."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = b.sample(50, mode='adaptive', tolerance=1.0)
        self.assertEqual(pts.shape, (50, 2))

    def test_adaptive_endpoints(self):
        u"""Premier et dernier points = P0 et Pn."""
        b = Bezier([[0, 0], [0, 100], [100, 100], [100, 0]])
        pts = b.sample(30, mode='adaptive', tolerance=0.5)
        np.testing.assert_allclose(pts[0], [0, 0], atol=0.5)
        np.testing.assert_allclose(pts[-1], [100, 0], atol=0.5)

    def test_adaptive_denser_at_curvature(self):
        u"""Plus de points la ou la courbure est forte."""
        # Courbe en S : courbure forte aux extremites, faible au milieu
        b = Bezier([[0, 0], [0, 100], [100, 0], [100, 100]])
        pts = b.sample(40, mode='adaptive', tolerance=0.5)
        # Mesurer la densite locale (inverse de l'espacement)
        ds = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        # Les extremites (debut et fin) doivent avoir des pas plus petits
        # que le milieu
        ds_start = np.mean(ds[:5])
        ds_mid = np.mean(ds[15:25])
        self.assertLess(ds_start, ds_mid)

    def test_adaptive_c1_spacing(self):
        u"""Variation d'espacement C1 : differences de pas lisses."""
        b = Bezier([[0, 0], [0, 100], [100, 100], [100, 0]])
        pts = b.sample(60, mode='adaptive', tolerance=0.5)
        ds = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        # Variation du pas : d(ds)/di
        dds = np.diff(ds)
        # Variation de la variation : d2(ds)/di2
        ddds = np.diff(dds)
        # La variation doit etre lisse (pas de sauts brusques)
        # Ratio max de d2ds par rapport a la plage de ds
        ds_range = ds.max() - ds.min()
        if ds_range > 1e-10:
            smoothness = np.max(np.abs(ddds)) / ds_range
            self.assertLess(smoothness, 0.5)

    def test_adaptive_no_tolerance_raises(self):
        u"""Mode adaptive sans tolerance -> ValueError."""
        b = Bezier([[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            b.sample(10, mode='adaptive')

    def test_adaptive_line_falls_back(self):
        u"""Droite : mode adaptive fonctionne (courbure ~0)."""
        b = Bezier([[0, 0], [100, 0]])
        pts = b.sample(11, mode='adaptive', tolerance=1.0)
        self.assertEqual(pts.shape, (11, 2))
        # Doit rester quasi uniforme
        ds = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        ratio = ds.max() / ds.min()
        self.assertLess(ratio, 2.0)

    def test_invalid_mode_raises(self):
        u"""Mode inconnu -> ValueError."""
        b = Bezier([[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            b.sample(10, mode='unknown')


class TestBezierTranslate(unittest.TestCase):
    u"""Tests de la translation."""

    def test_translate_moves_all_cpoints(self):
        u"""Translation deplace tous les points de controle."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        b.translate(10, -5)
        np.testing.assert_allclose(b.start_cpoint, [10, -5], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [410, -5], atol=1e-10)

    def test_translate_curve_shifted(self):
        u"""La courbe evaluee est decalee."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        t = np.linspace(0, 1, 20)
        pts_before = b.evaluate(t)
        b.translate(50, -30)
        pts_after = b.evaluate(t)
        np.testing.assert_allclose(pts_after, pts_before + [50, -30], atol=1e-10)

    def test_translate_returns_self(self):
        u"""translate retourne self (chainage)."""
        b = Bezier([[0, 0], [1, 1]])
        self.assertIs(b.translate(1, 2), b)

    def test_translate_invalidates_cache(self):
        u"""translate invalide le cache."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.points
        self.assertIn('points', b._cache)
        b.translate(1, 1)
        self.assertEqual(len(b._cache), 0)

    def test_translate_zero(self):
        u"""Translation nulle : courbe inchangee."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts_before = b.control_points.copy()
        b.translate(0, 0)
        np.testing.assert_allclose(b.control_points, pts_before, atol=1e-10)


class TestBezierRotate(unittest.TestCase):
    u"""Tests de la rotation."""

    def test_rotate_90_deg(self):
        u"""Rotation 90 deg antihoraire autour de P0."""
        b = Bezier([[0, 0], [100, 0]])
        b.rotate(90.0)
        np.testing.assert_allclose(b.start_cpoint, [0, 0], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [0, 100], atol=1e-10)

    def test_rotate_180_deg(self):
        u"""Rotation 180 deg autour de P0."""
        b = Bezier([[0, 0], [100, 50], [200, 0]])
        b.rotate(180.0)
        np.testing.assert_allclose(b.start_cpoint, [0, 0], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [-200, 0], atol=1e-10)

    def test_rotate_360_identity(self):
        u"""Rotation 360 deg : retour a l'identique."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts_before = b.control_points.copy()
        b.rotate(360.0)
        np.testing.assert_allclose(b.control_points, pts_before, atol=1e-10)

    def test_rotate_custom_center(self):
        u"""Rotation autour d'un centre personnalise."""
        b = Bezier([[100, 0], [200, 0]])
        b.rotate(90.0, center=[100, 0])
        np.testing.assert_allclose(b.start_cpoint, [100, 0], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [100, 100], atol=1e-10)

    def test_rotate_preserves_shape(self):
        u"""La forme de la courbe est preservee apres rotation."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        t = np.linspace(0, 1, 30)
        pts_before = b.evaluate(t)
        b.rotate(45.0)
        pts_after = b.evaluate(t)
        # Les distances inter-points sont preservees
        d_before = np.sqrt(np.sum(np.diff(pts_before, axis=0)**2, axis=1))
        d_after = np.sqrt(np.sum(np.diff(pts_after, axis=0)**2, axis=1))
        np.testing.assert_allclose(d_after, d_before, atol=1e-10)

    def test_rotate_returns_self(self):
        u"""rotate retourne self (chainage)."""
        b = Bezier([[0, 0], [1, 1]])
        self.assertIs(b.rotate(10), b)

    def test_rotate_invalidates_cache(self):
        u"""rotate invalide le cache."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.points
        self.assertIn('points', b._cache)
        b.rotate(10)
        self.assertEqual(len(b._cache), 0)


class TestBezierScale(unittest.TestCase):
    u"""Tests du scaling."""

    def test_scale_double(self):
        u"""Scale x2 depuis P0 : Pn a distance double."""
        b = Bezier([[0, 0], [100, 200], [400, 0]])
        b.scale(2.0)
        np.testing.assert_allclose(b.start_cpoint, [0, 0], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [800, 0], atol=1e-10)

    def test_scale_half(self):
        u"""Scale x0.5 depuis P0."""
        b = Bezier([[0, 0], [100, 200], [400, 0]])
        b.scale(0.5)
        np.testing.assert_allclose(b.end_cpoint, [200, 0], atol=1e-10)

    def test_scale_custom_center(self):
        u"""Scale depuis un centre personnalise."""
        b = Bezier([[0, 0], [100, 0]])
        b.scale(2.0, center=[50, 0])
        np.testing.assert_allclose(b.start_cpoint, [-50, 0], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [150, 0], atol=1e-10)

    def test_scale_1_identity(self):
        u"""Scale x1 : courbe inchangee."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts_before = b.control_points.copy()
        b.scale(1.0)
        np.testing.assert_allclose(b.control_points, pts_before, atol=1e-10)

    def test_scale_negative(self):
        u"""Scale negatif : miroir."""
        b = Bezier([[0, 0], [100, 0]])
        b.scale(-1.0)
        np.testing.assert_allclose(b.start_cpoint, [0, 0], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [-100, 0], atol=1e-10)

    def test_scale_returns_self(self):
        u"""scale retourne self (chainage)."""
        b = Bezier([[0, 0], [1, 1]])
        self.assertIs(b.scale(2), b)

    def test_scale_invalidates_cache(self):
        u"""scale invalide le cache."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.points
        self.assertIn('points', b._cache)
        b.scale(2.0)
        self.assertEqual(len(b._cache), 0)

    def test_chaining(self):
        u"""Chainage translate + rotate + scale."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        result = b.translate(10, 20).rotate(45).scale(0.5)
        self.assertIs(result, b)


class TestBezierReverse(unittest.TestCase):
    u"""Tests de la methode reverse."""

    def test_reverse_swaps_endpoints(self):
        u"""reverse echange P0 et Pn."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        p0_before = b.start_cpoint.copy()
        pn_before = b.end_cpoint.copy()
        b.reverse()
        np.testing.assert_array_equal(b.start_cpoint, pn_before)
        np.testing.assert_array_equal(b.end_cpoint, p0_before)

    def test_reverse_preserves_geometry(self):
        u"""reverse ne change pas la geometrie, seulement le sens."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts_before = b.sample(50)
        b.reverse()
        pts_after = b.sample(50)
        # Les points sont les memes mais en ordre inverse
        np.testing.assert_allclose(pts_after, pts_before[::-1], atol=1e-10)

    def test_reverse_returns_self(self):
        u"""reverse retourne self (chainage)."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        result = b.reverse()
        self.assertIs(result, b)

    def test_reverse_invalidates_cache(self):
        u"""reverse invalide le cache."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.points
        self.assertIn('points', b._cache)
        b.reverse()
        self.assertEqual(len(b._cache), 0)

    def test_double_reverse_identity(self):
        u"""Deux reverse successifs = identite."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        cpts_orig = b.control_points.copy()
        b.reverse().reverse()
        np.testing.assert_array_equal(b.control_points, cpts_orig)


class TestBezierApproximate(unittest.TestCase):
    u"""Tests de la methode approximate et du constructeur avec degree."""

    def test_fit_known_cubic(self):
        u"""Points echantillonnes d'une cubique : bonne approximation."""
        b_orig = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = b_orig.sample(50)
        b_fit = Bezier([[0, 0], [1, 1]])
        b_fit.approximate(pts, degree=3)
        t = np.linspace(0, 1, 100)
        err = np.max(np.sqrt(np.sum(
            (b_fit.evaluate(t) - b_orig.evaluate(t))**2, axis=1)))
        self.assertLess(err, 5.0)

    def test_fit_known_quadratic(self):
        u"""Points echantillonnes d'une quadratique : reconstruction fidele."""
        b_orig = Bezier([[0, 0], [100, 200], [200, 0]])
        pts = b_orig.sample(30)
        b_fit = Bezier(pts, degree=2)
        t = np.linspace(0, 1, 50)
        err = np.max(np.sqrt(np.sum(
            (b_fit.evaluate(t) - b_orig.evaluate(t))**2, axis=1)))
        self.assertLess(err, 1.0)

    def test_clamp_ends_true(self):
        u"""clamp_ends=True : extremites exactes."""
        pts = np.array([[10, 20], [50, 80], [100, 120],
                        [200, 100], [300, 50], [400, -10]], dtype=float)
        b = Bezier([[0, 0], [1, 1]])
        b.approximate(pts, degree=3, clamp_ends=True)
        np.testing.assert_allclose(b.start_cpoint, [10, 20], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [400, -10], atol=1e-10)

    def test_clamp_ends_false(self):
        u"""clamp_ends=False : fit libre sans contrainte."""
        pts = np.array([[0, 0], [50, 80], [100, 120],
                        [200, 100], [300, 50], [400, 0]], dtype=float)
        b = Bezier([[0, 0], [1, 1]])
        b.approximate(pts, degree=3, clamp_ends=False)
        self.assertEqual(b.degree, 3)

    def test_init_with_degree(self):
        u"""Constructeur avec degree : cree une courbe ajustee."""
        b_orig = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = b_orig.sample(50)
        b = Bezier(pts, degree=3, name='Ajustee')
        self.assertEqual(b.degree, 3)
        self.assertEqual(b.name, 'Ajustee')
        t = np.linspace(0, 1, 50)
        err = np.max(np.sqrt(np.sum(
            (b.evaluate(t) - b_orig.evaluate(t))**2, axis=1)))
        self.assertLess(err, 5.0)

    def test_degree_1_line(self):
        u"""Degre 1 clampe : droite entre premier et dernier point."""
        pts = np.array([[0, 0], [50, 30], [100, 55], [200, 100]], dtype=float)
        b = Bezier(pts, degree=1)
        np.testing.assert_allclose(b.start_cpoint, [0, 0], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [200, 100], atol=1e-10)
        self.assertEqual(b.degree, 1)

    def test_default_degree_uses_current(self):
        u"""Sans degree explicite : utilise le degre courant."""
        b = Bezier([[0, 0], [50, 100], [100, 100], [150, 0]])
        pts = np.array([[0, 0], [30, 60], [80, 90], [120, 80],
                        [150, 40], [180, 0]], dtype=float)
        b.approximate(pts)
        self.assertEqual(b.degree, 3)

    def test_approximate_returns_self(self):
        u"""approximate retourne self (chainage)."""
        b = Bezier([[0, 0], [1, 1]])
        pts = np.array([[0, 0], [50, 80], [100, 0]], dtype=float)
        self.assertIs(b.approximate(pts, degree=2), b)

    def test_approximate_invalidates_cache(self):
        u"""approximate invalide le cache."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.points
        self.assertIn('points', b._cache)
        pts = np.array([[0, 0], [50, 80], [100, 120],
                        [200, 0]], dtype=float)
        b.approximate(pts, degree=3)
        self.assertEqual(len(b._cache), 0)

    def test_too_few_points_raises(self):
        u"""Pas assez de points pour le degre -> ValueError."""
        b = Bezier([[0, 0], [1, 1]])
        pts = np.array([[0, 0], [100, 0]], dtype=float)
        with self.assertRaises(ValueError):
            b.approximate(pts, degree=3)

    def test_invalid_degree_raises(self):
        u"""degree < 1 -> ValueError."""
        b = Bezier([[0, 0], [1, 1]])
        pts = np.array([[0, 0], [50, 80], [100, 0]], dtype=float)
        with self.assertRaises(ValueError):
            b.approximate(pts, degree=0)

    def test_higher_degree_fit(self):
        u"""Fit de degre 5 : bonne approximation d'une quartique."""
        b_orig = Bezier([[0, 0], [50, 200], [200, 200],
                         [350, -50], [400, 0]])
        pts = b_orig.sample(60)
        b_fit = Bezier(pts, degree=5)
        t = np.linspace(0, 1, 50)
        err = np.max(np.sqrt(np.sum(
            (b_fit.evaluate(t) - b_orig.evaluate(t))**2, axis=1)))
        self.assertLess(err, 35.0)

    def test_bernstein_matrix_endpoints(self):
        u"""Matrice de Bernstein : B(0) = [1,0,...,0], B(1) = [0,...,0,1]."""
        t = np.array([0.0, 0.5, 1.0])
        N = Bezier._bernstein_matrix(3, t)
        # t=0 : seul B_{0,3}(0) = 1
        np.testing.assert_allclose(N[0], [1, 0, 0, 0], atol=1e-15)
        # t=1 : seul B_{3,3}(1) = 1
        np.testing.assert_allclose(N[2], [0, 0, 0, 1], atol=1e-15)

    def test_bernstein_partition_of_unity(self):
        u"""Somme des Bernstein = 1 pour tout t."""
        t = np.linspace(0, 1, 20)
        N = Bezier._bernstein_matrix(4, t)
        np.testing.assert_allclose(np.sum(N, axis=1), 1.0, atol=1e-14)

    def test_degenerate_points(self):
        u"""Points tous confondus : courbe degeneree sans erreur."""
        pts = np.array([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=float)
        b = Bezier(pts, degree=2)
        np.testing.assert_allclose(b.evaluate(0.5), [5, 5], atol=1e-10)

    def test_max_iter_zero_no_reprojection(self):
        u"""max_iter=0 : parametrisation corde seule, pas de reprojection."""
        b_orig = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = b_orig.sample(50)
        b_no_iter = Bezier([[0, 0], [1, 1]])
        b_no_iter.approximate(pts, degree=3, max_iter=0)
        b_iter = Bezier([[0, 0], [1, 1]])
        b_iter.approximate(pts, degree=3, max_iter=5)
        t = np.linspace(0, 1, 50)
        err_no = np.max(np.sqrt(np.sum(
            (b_no_iter.evaluate(t) - b_orig.evaluate(t))**2, axis=1)))
        err_it = np.max(np.sqrt(np.sum(
            (b_iter.evaluate(t) - b_orig.evaluate(t))**2, axis=1)))
        # Avec iterations, l'erreur est nettement plus faible
        self.assertLess(err_it, err_no * 0.5)


class TestBezierElevate(unittest.TestCase):
    u"""Tests de l'elevation de degre."""

    def test_degree_increases(self):
        u"""Elevation augmente le degre de 1."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        self.assertEqual(b.degree, 3)
        b.elevate()
        self.assertEqual(b.degree, 4)
        self.assertEqual(len(b.control_points), 5)

    def test_curve_unchanged(self):
        u"""La courbe evaluee reste identique apres elevation."""
        b1 = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        t = np.linspace(0, 1, 50)
        pts_before = b1.evaluate(t)
        b1.elevate()
        pts_after = b1.evaluate(t)
        np.testing.assert_allclose(pts_after, pts_before, atol=1e-10)

    def test_endpoints_preserved(self):
        u"""P0 et Pn inchanges apres elevation."""
        b = Bezier([[10, 20], [50, 100], [90, 100], [130, 20]])
        b.elevate()
        np.testing.assert_allclose(b.start_cpoint, [10, 20], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [130, 20], atol=1e-10)

    def test_elevate_line(self):
        u"""Droite elevee : courbe toujours droite."""
        b = Bezier([[0, 0], [100, 50]])
        b.elevate()
        self.assertEqual(b.degree, 2)
        # Milieu du polygone de controle = milieu de la droite
        np.testing.assert_allclose(b.cpoint(1), [50, 25], atol=1e-10)

    def test_elevate_multiple(self):
        u"""Elevation multiple (times=3) : degre +3, courbe identique."""
        b = Bezier([[0, 0], [50, 100], [100, 0]])
        t = np.linspace(0, 1, 30)
        pts_before = b.evaluate(t)
        b.elevate(times=3)
        self.assertEqual(b.degree, 5)
        pts_after = b.evaluate(t)
        np.testing.assert_allclose(pts_after, pts_before, atol=1e-10)

    def test_elevate_returns_self(self):
        u"""elevate retourne self (chainage)."""
        b = Bezier([[0, 0], [1, 1]])
        result = b.elevate()
        self.assertIs(result, b)

    def test_elevate_invalidates_cache(self):
        u"""elevate invalide le cache."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.points
        self.assertIn('points', b._cache)
        b.elevate()
        self.assertEqual(len(b._cache), 0)

    def test_elevate_times_invalid(self):
        u"""times < 1 -> ValueError."""
        b = Bezier([[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            b.elevate(times=0)


class TestBezierReduce(unittest.TestCase):
    u"""Tests de la reduction de degre."""

    def test_degree_decreases(self):
        u"""Reduction diminue le degre de 1."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        self.assertEqual(b.degree, 3)
        b.reduce()
        self.assertEqual(b.degree, 2)
        self.assertEqual(len(b.control_points), 3)

    def test_elevate_then_reduce_recovers(self):
        u"""Elevate puis reduce : retrouve la courbe originale."""
        pts_orig = np.array([[0, 0], [100, 200], [300, 200], [400, 0]],
                            dtype=float)
        b = Bezier(pts_orig.copy())
        t = np.linspace(0, 1, 50)
        pts_before = b.evaluate(t)
        b.elevate()
        b.reduce()
        pts_after = b.evaluate(t)
        np.testing.assert_allclose(pts_after, pts_before, atol=0.1)

    def test_endpoints_preserved(self):
        u"""clamp_ends=True : P0 et Pn conserves."""
        b = Bezier([[10, 20], [50, 100], [90, 100], [130, 20]])
        b.reduce()
        np.testing.assert_allclose(b.start_cpoint, [10, 20], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [130, 20], atol=1e-10)

    def test_tangents_preserved(self):
        u"""clamp_tangents=True : derivees aux extremites conservees."""
        P = np.array([[0, 0], [30, 90], [70, 90], [100, 0]], dtype=float)
        b_orig = Bezier(P.copy())
        d0_orig = b_orig.derivative(0.0)
        d1_orig = b_orig.derivative(1.0)
        # Elever puis reduire pour avoir un cas propre
        b_orig.elevate(times=2)
        b_orig.reduce(times=2, clamp_tangents=True)
        d0_red = b_orig.derivative(0.0)
        d1_red = b_orig.derivative(1.0)
        np.testing.assert_allclose(d0_red, d0_orig, atol=1.0)
        np.testing.assert_allclose(d1_red, d1_orig, atol=1.0)

    def test_no_clamp_ends(self):
        u"""clamp_ends=False : extremites non garanties."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        b.elevate(times=3)
        b.reduce(clamp_ends=False, clamp_tangents=False)
        # Pas d'assertion stricte, juste verifier que ca tourne
        self.assertEqual(b.degree, 5)

    def test_clamp_tangents_implies_ends(self):
        u"""clamp_tangents=True force clamp_ends=True."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        b.elevate()
        b.reduce(clamp_ends=False, clamp_tangents=True)
        # Extremites preservees malgre clamp_ends=False
        np.testing.assert_allclose(b.start_cpoint, [0, 0], atol=1e-10)
        np.testing.assert_allclose(b.end_cpoint, [400, 0], atol=1e-10)

    def test_reduce_multiple(self):
        u"""Reduction multiple (times=2)."""
        b = Bezier([[0, 0], [50, 100], [100, 0]])
        b.elevate(times=3)
        self.assertEqual(b.degree, 5)
        b.reduce(times=2)
        self.assertEqual(b.degree, 3)

    def test_reduce_below_1_raises(self):
        u"""Reduire en dessous du degre 1 -> ValueError."""
        b = Bezier([[0, 0], [100, 50]])  # degre 1
        with self.assertRaises(ValueError):
            b.reduce()

    def test_reduce_times_invalid(self):
        u"""times < 1 -> ValueError."""
        b = Bezier([[0, 0], [100, 200], [200, 0]])
        with self.assertRaises(ValueError):
            b.reduce(times=0)

    def test_reduce_returns_self(self):
        u"""reduce retourne self (chainage)."""
        b = Bezier([[0, 0], [50, 100], [100, 100], [150, 0]])
        result = b.reduce()
        self.assertIs(result, b)

    def test_reduce_invalidates_cache(self):
        u"""reduce invalide le cache."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.points
        self.assertIn('points', b._cache)
        b.reduce()
        self.assertEqual(len(b._cache), 0)

    def test_reduce_cubic_to_quadratic_shape(self):
        u"""Cubique symetrique -> quadratique : forme approximee."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        t = np.linspace(0, 1, 20)
        pts_orig = b.evaluate(t)
        b.reduce()
        pts_red = b.evaluate(t)
        # Erreur max raisonnable pour cette courbe
        err = np.max(np.sqrt(np.sum((pts_red - pts_orig)**2, axis=1)))
        self.assertLess(err, 30.0)


class TestBezierTangentsProperty(unittest.TestCase):
    u"""Tests de la property tangents et sample_tangents."""

    def test_tangents_shape(self):
        u"""tangents retourne ndarray(n_points, 2)."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        self.assertEqual(b.tangents.shape, (100, 2))

    def test_tangents_unit_norm(self):
        u"""Toutes les tangentes sont de norme 1."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]], n_points=30)
        norms = np.linalg.norm(b.tangents, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_tangents_cached(self):
        u"""tangents est mis en cache."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.tangents
        self.assertIn('tangents', b._cache)

    def test_tangents_returns_copy(self):
        u"""tangents retourne une copie."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        t1 = b.tangents
        t1[0] = [999, 999]
        t2 = b.tangents
        self.assertFalse(np.allclose(t2[0], [999, 999]))

    def test_tangents_invalidated_by_geometry(self):
        u"""translate_cpoint invalide le cache tangents."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.tangents
        b.translate_cpoint(1, [1, 1])
        self.assertNotIn('tangents', b._cache)

    def test_tangents_invalidated_by_n_points(self):
        u"""Changer n_points invalide tangents."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.tangents
        b.n_points = 50
        self.assertNotIn('tangents', b._cache)

    def test_tangents_consistent_with_points(self):
        u"""tangents a la meme taille que points."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]], n_points=40)
        self.assertEqual(len(b.tangents), len(b.points))

    def test_tangents_same_sampling_as_points(self):
        u"""tangents evaluees aux memes t que points."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]], n_points=30)
        t = b._sample_t_values(30, 'curvilinear')
        tg_manual = b.tangent(t)
        np.testing.assert_allclose(b.tangents, tg_manual, atol=1e-10)

    def test_sample_tangents_custom(self):
        u"""sample_tangents avec params personnalises."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        tg, t = b.sample_tangents(30)
        self.assertEqual(tg.shape, (30, 2))
        self.assertEqual(t.shape, (30,))
        self.assertAlmostEqual(t[0], 0.0, places=10)
        self.assertAlmostEqual(t[-1], 1.0, places=10)


class TestBezierNormalsProperty(unittest.TestCase):
    u"""Tests de la property normals et sample_normals."""

    def test_normals_shape(self):
        u"""normals retourne ndarray(n_points, 2)."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        self.assertEqual(b.normals.shape, (100, 2))

    def test_normals_unit_norm(self):
        u"""Toutes les normales sont de norme 1."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]], n_points=30)
        norms = np.linalg.norm(b.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_normals_orthogonal_to_tangents(self):
        u"""Normales orthogonales aux tangentes."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]], n_points=30)
        dots = np.sum(b.tangents * b.normals, axis=1)
        np.testing.assert_allclose(dots, 0.0, atol=1e-10)

    def test_normals_cached(self):
        u"""normals est mis en cache."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.normals
        self.assertIn('normals', b._cache)

    def test_normals_returns_copy(self):
        u"""normals retourne une copie."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        n1 = b.normals
        n1[0] = [999, 999]
        n2 = b.normals
        self.assertFalse(np.allclose(n2[0], [999, 999]))

    def test_normals_invalidated_by_geometry(self):
        u"""translate_cpoint invalide le cache normals."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.normals
        b.translate_cpoint(1, [1, 1])
        self.assertNotIn('normals', b._cache)

    def test_normals_invalidated_by_n_points(self):
        u"""Changer n_points invalide normals."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.normals
        b.n_points = 50
        self.assertNotIn('normals', b._cache)

    def test_normals_same_sampling_as_points(self):
        u"""normals evaluees aux memes t que points."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]], n_points=30)
        t = b._sample_t_values(30, 'curvilinear')
        nm_manual = b.normal(t)
        np.testing.assert_allclose(b.normals, nm_manual, atol=1e-10)

    def test_sample_normals_custom(self):
        u"""sample_normals avec params personnalises."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        nm, t = b.sample_normals(25)
        self.assertEqual(nm.shape, (25, 2))
        self.assertEqual(t.shape, (25,))
        self.assertAlmostEqual(t[0], 0.0, places=10)
        self.assertAlmostEqual(t[-1], 1.0, places=10)


class TestBezierCurvatures(unittest.TestCase):
    u"""Tests de la property curvatures et sample_curvatures."""

    def test_curvatures_shape(self):
        u"""curvatures retourne ndarray(n_points,)."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        k = b.curvatures
        self.assertEqual(k.shape, (100,))

    def test_curvatures_custom_n(self):
        u"""n_points personnalise."""
        b = Bezier([[0, 0], [100, 200], [400, 0]], n_points=50)
        self.assertEqual(b.curvatures.shape, (50,))

    def test_curvatures_line_zero(self):
        u"""Droite : toutes les courbures proches de 0."""
        b = Bezier([[0, 0], [100, 0]], n_points=20)
        np.testing.assert_allclose(b.curvatures, 0.0, atol=1e-10)

    def test_curvatures_nonzero(self):
        u"""Cubique : courbures non nulles au milieu."""
        b = Bezier([[0, 0], [0, 100], [100, 100], [100, 0]], n_points=20)
        self.assertTrue(np.any(np.abs(b.curvatures) > 0.001))

    def test_curvatures_cached(self):
        u"""curvatures est mis en cache."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.curvatures
        self.assertIn('curvatures', b._cache)

    def test_curvatures_returns_copy(self):
        u"""curvatures retourne une copie."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        k1 = b.curvatures
        k1[0] = 999.0
        k2 = b.curvatures
        self.assertNotAlmostEqual(k2[0], 999.0)

    def test_curvatures_invalidated_by_geometry(self):
        u"""translate_cpoint invalide le cache curvatures."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.curvatures
        self.assertIn('curvatures', b._cache)
        b.translate_cpoint(1, [1, 1])
        self.assertNotIn('curvatures', b._cache)

    def test_curvatures_invalidated_by_n_points(self):
        u"""Changer n_points invalide curvatures."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        _ = b.curvatures
        self.assertIn('curvatures', b._cache)
        b.n_points = 50
        self.assertNotIn('curvatures', b._cache)

    def test_curvatures_consistent_with_points(self):
        u"""curvatures a la meme taille que points."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]], n_points=40)
        self.assertEqual(len(b.curvatures), len(b.points))

    def test_sample_curvatures_custom(self):
        u"""sample_curvatures avec params personnalises."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        k, t = b.sample_curvatures(30)
        self.assertEqual(k.shape, (30,))
        self.assertEqual(t.shape, (30,))
        self.assertAlmostEqual(t[0], 0.0, places=10)
        self.assertAlmostEqual(t[-1], 1.0, places=10)

    def test_sample_curvatures_adaptive(self):
        u"""sample_curvatures en mode adaptive."""
        b = Bezier([[0, 0], [0, 100], [100, 100], [100, 0]])
        k, t = b.sample_curvatures(40, mode='adaptive', tolerance=1.0)
        self.assertEqual(k.shape, (40,))
        self.assertEqual(t.shape, (40,))

    def test_curvatures_same_sampling_as_points(self):
        u"""curvatures evaluees aux memes t que points."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]], n_points=30)
        # Evaluer la courbure manuellement aux memes t
        t = b._sample_t_values(30, 'curvilinear')
        k_manual = b.curvature(t)
        np.testing.assert_allclose(b.curvatures, k_manual, atol=1e-10)


class TestBezierCache(unittest.TestCase):
    u"""Tests du mecanisme de cache."""

    def setUp(self):
        self.bez = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])

    def test_points_cached(self):
        u"""Deuxieme appel a points utilise le cache."""
        _ = self.bez.points
        self.assertIn('points', self.bez._cache)

    def test_points_consistent(self):
        u"""Appels successifs retournent le meme resultat."""
        p1 = self.bez.points
        p2 = self.bez.points
        np.testing.assert_array_equal(p1, p2)

    def test_points_returns_copy(self):
        u"""points retourne une copie (modification sans corrompre le cache)."""
        p1 = self.bez.points
        p1[0] = [999, 999]
        p2 = self.bez.points
        self.assertFalse(np.allclose(p2[0], [999, 999]))

    def test_translate_cpoint_invalidates(self):
        u"""translate_cpoint invalide tout le cache."""
        _ = self.bez.points
        self.assertIn('points', self.bez._cache)
        self.bez.translate_cpoint(1, [1, 1])
        self.assertEqual(len(self.bez._cache), 0)

    def test_n_points_setter_invalidates(self):
        u"""Changer n_points invalide le cache points."""
        _ = self.bez.points
        self.assertIn('points', self.bez._cache)
        self.bez.n_points = 50
        self.assertNotIn('points', self.bez._cache)

    def test_sample_mode_setter_invalidates(self):
        u"""Changer sample_mode invalide le cache points."""
        _ = self.bez.points
        self.assertIn('points', self.bez._cache)
        self.bez.sample_mode = 'adaptive'
        self.assertNotIn('points', self.bez._cache)

    def test_tolerance_setter_invalidates(self):
        u"""Changer tolerance invalide le cache points."""
        _ = self.bez.points
        self.assertIn('points', self.bez._cache)
        self.bez.tolerance = 1.0
        self.assertNotIn('points', self.bez._cache)

    def test_same_value_no_invalidation(self):
        u"""Setter avec meme valeur ne vide pas le cache."""
        _ = self.bez.points
        self.assertIn('points', self.bez._cache)
        self.bez.n_points = 100  # meme valeur
        self.assertIn('points', self.bez._cache)

    def test_arc_length_cached(self):
        u"""_arc_length_table est mise en cache."""
        self.bez._arc_length_table(500)
        self.assertIn(('arc_length', 500), self.bez._cache)

    def test_geometry_invalidates_arc_length(self):
        u"""translate_cpoint invalide aussi le cache arc_length."""
        self.bez._arc_length_table(500)
        self.assertIn(('arc_length', 500), self.bez._cache)
        self.bez.translate_cpoint(0, [1, 0])
        self.assertNotIn(('arc_length', 500), self.bez._cache)

    def test_sampling_preserves_arc_length(self):
        u"""Changer n_points ne vide pas le cache arc_length."""
        _ = self.bez.points  # remplit arc_length + points
        has_arc = any(k[0] == 'arc_length' for k in self.bez._cache
                      if isinstance(k, tuple))
        self.assertTrue(has_arc)
        self.bez.n_points = 50  # invalide points, pas arc_length
        has_arc = any(k[0] == 'arc_length' for k in self.bez._cache
                      if isinstance(k, tuple))
        self.assertTrue(has_arc)


class TestBezierMaxDeviation(unittest.TestCase):
    u"""Tests de la methode max_deviation."""

    def test_exact_curve_zero_deviation(self):
        u"""Points echantillonnes de la courbe : deviation ~0."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = b.sample(50)
        dev, t_max = b.max_deviation(pts)
        self.assertLess(dev, 0.1)
        self.assertGreaterEqual(t_max, 0.0)
        self.assertLessEqual(t_max, 1.0)

    def test_line_deviation(self):
        u"""Droite : points alignes -> deviation 0."""
        b = Bezier([[0, 0], [100, 0]])
        pts = np.array([[0, 0], [25, 0], [50, 0], [75, 0], [100, 0]])
        dev, t_max = b.max_deviation(pts)
        self.assertLess(dev, 1e-10)

    def test_offset_points_deviation(self):
        u"""Points decales de la courbe : deviation > 0."""
        b = Bezier([[0, 0], [100, 0]])
        pts = np.array([[0, 0], [50, 10], [100, 0]])
        dev, t_max = b.max_deviation(pts)
        self.assertAlmostEqual(dev, 10.0, delta=0.5)
        # Le point le plus eloigne est (50, 10), projete a t~0.5
        self.assertAlmostEqual(t_max, 0.5, delta=0.05)

    def test_returns_tuple(self):
        u"""max_deviation retourne un tuple (float, float)."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = b.sample(20)
        result = b.max_deviation(pts)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_t_max_in_range(self):
        u"""t_max est dans [0, 1]."""
        b = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = np.array([[0, 0], [200, 150], [400, 0]], dtype=float)
        dev, t_max = b.max_deviation(pts)
        self.assertGreaterEqual(t_max, 0.0)
        self.assertLessEqual(t_max, 1.0)

    def test_invalid_shape_raises(self):
        u"""Shape invalide -> ValueError."""
        b = Bezier([[0, 0], [100, 0]])
        with self.assertRaises(ValueError):
            b.max_deviation(np.array([1, 2, 3]))


class TestBezierFindDegree(unittest.TestCase):
    u"""Tests du mode degree='find'."""

    def test_find_cubic_from_samples(self):
        u"""Points d'une cubique : degree='find' trouve degre 3."""
        b_orig = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = b_orig.sample(50)
        b = Bezier([[0, 0], [1, 1]])
        b.approximate(pts, degree='find', max_dev=1.0)
        self.assertLessEqual(b.degree, 5)
        dev, _t = b.max_deviation(pts)
        self.assertLessEqual(dev, 1.0)

    def test_find_line(self):
        u"""Points quasi-alignes : degre 1 suffit."""
        pts = np.array([[0, 0], [100, 0], [200, 0], [300, 0]], dtype=float)
        b = Bezier([[0, 0], [1, 1]])
        b.approximate(pts, degree='find', max_dev=1.0)
        self.assertEqual(b.degree, 1)

    def test_find_respects_max_dev(self):
        u"""Le degre trouve respecte la contrainte max_dev."""
        b_orig = Bezier([[0, 0], [50, 200], [200, 200],
                         [350, -50], [400, 0]])
        pts = b_orig.sample(60)
        b = Bezier([[0, 0], [1, 1]])
        b.approximate(pts, degree='find', max_dev=5.0)
        dev, _t = b.max_deviation(pts)
        self.assertLessEqual(dev, 5.0)

    def test_find_tight_tolerance(self):
        u"""Tolerance serree : degre plus eleve."""
        b_orig = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = b_orig.sample(50)
        b_loose = Bezier([[0, 0], [1, 1]])
        b_loose.approximate(pts, degree='find', max_dev=50.0)
        b_tight = Bezier([[0, 0], [1, 1]])
        b_tight.approximate(pts, degree='find', max_dev=0.1)
        self.assertGreaterEqual(b_tight.degree, b_loose.degree)

    def test_find_via_constructor(self):
        u"""Constructeur avec degree='find'."""
        b_orig = Bezier([[0, 0], [100, 200], [300, 200], [400, 0]])
        pts = b_orig.sample(50)
        b = Bezier(pts, degree='find', max_dev=1.0, name='Auto')
        self.assertEqual(b.name, 'Auto')
        dev, _t = b.max_deviation(pts)
        self.assertLessEqual(dev, 1.0)

    def test_find_no_max_dev_raises(self):
        u"""degree='find' sans max_dev -> ValueError."""
        b = Bezier([[0, 0], [1, 1]])
        pts = np.array([[0, 0], [50, 80], [100, 0]], dtype=float)
        with self.assertRaises(ValueError):
            b.approximate(pts, degree='find')

    def test_find_negative_max_dev_raises(self):
        u"""degree='find' avec max_dev <= 0 -> ValueError."""
        b = Bezier([[0, 0], [1, 1]])
        pts = np.array([[0, 0], [50, 80], [100, 0]], dtype=float)
        with self.assertRaises(ValueError):
            b.approximate(pts, degree='find', max_dev=-1.0)

    def test_find_returns_self(self):
        u"""degree='find' retourne self (chainage)."""
        b = Bezier([[0, 0], [1, 1]])
        pts = np.array([[0, 0], [50, 80], [100, 0]], dtype=float)
        result = b.approximate(pts, degree='find', max_dev=50.0)
        self.assertIs(result, b)

    def test_find_max_degree_reached(self):
        u"""Tolerance impossible : atteint le degre max sans erreur."""
        pts = np.array([[0, 0], [30, 50], [70, -20], [100, 40],
                         [150, 10], [200, 0]], dtype=float)
        b = Bezier([[0, 0], [1, 1]])
        b.approximate(pts, degree='find', max_dev=0.001)
        # Pas d'exception, degre <= m
        self.assertLessEqual(b.degree, len(pts) - 1)

    def test_find_quadratic_exact(self):
        u"""Points d'une quadratique : degre 2 suffit."""
        b_orig = Bezier([[0, 0], [100, 200], [200, 0]])
        pts = b_orig.sample(30)
        b = Bezier(pts, degree='find', max_dev=1.0)
        self.assertLessEqual(b.degree, 3)
        dev, _t = b.max_deviation(pts)
        self.assertLessEqual(dev, 1.0)


if __name__ == '__main__':
    unittest.main()
