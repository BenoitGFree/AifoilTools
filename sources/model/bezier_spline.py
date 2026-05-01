#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Spline de Bezier composee de N segments de Bezier consecutifs.

Chaque jonction entre deux segments a un niveau de continuite :
C0 (positional), C1 (tangent) ou C2 (courbure).

Usage::

    from model.bezier import Bezier
    from model.bezier_spline import BezierSpline

    b1 = Bezier([[0,0], [100,200], [200,200], [300,0]])
    b2 = Bezier([[300,0], [400,-100], [500,0]])
    spline = BezierSpline([b1, b2], continuities=['C1'])

    # Parametrisation globale : t in [0, N]
    pt = spline.evaluate(1.0)   # point de jonction

@author: Nervures
@date: 2026-02
"""

import logging

import numpy as np

from .bezier import Bezier

logger = logging.getLogger(__name__)

VALID_CONTINUITIES = ('C0', 'C1', 'C2')


class BezierSpline(object):
    u"""Spline composee de N courbes de Bezier consecutives.

    Parametrisation globale : t dans [0, N] ou N = nombre de segments.
    Le segment k couvre l'intervalle [k, k+1].
    """

    def __init__(self, segments, continuities=None, name='Sans nom',
                 n_points=100, sample_mode='curvilinear',
                 tolerance=None):
        u"""
        :param segments: liste de courbes de Bezier consecutives
        :type segments: list[Bezier]
        :param continuities: niveau de continuite a chaque jonction.
            Longueur = len(segments) - 1.  Valeurs : 'C0', 'C1', 'C2'.
            Defaut : tout 'C0'.
        :type continuities: list[str] or None
        :param name: nom de la spline
        :type name: str
        :param n_points: nombre total de points d'echantillonnage
        :type n_points: int
        :param sample_mode: mode d'echantillonnage
        :type sample_mode: str
        :param tolerance: tolerance pour le mode adaptive
        :type tolerance: float or None
        """
        if not segments or len(segments) < 1:
            raise ValueError(u"Il faut au moins 1 segment")
        for i, s in enumerate(segments):
            if not isinstance(s, Bezier):
                raise TypeError(
                    u"segments[%d] n'est pas un Bezier" % i)

        self._segments = list(segments)
        n = len(self._segments)

        if continuities is None:
            self._continuities = ['C0'] * max(0, n - 1)
        else:
            if len(continuities) != n - 1:
                raise ValueError(
                    u"continuities doit avoir %d elements, "
                    u"recu %d" % (n - 1, len(continuities)))
            for i, c in enumerate(continuities):
                if c not in VALID_CONTINUITIES:
                    raise ValueError(
                        u"continuite[%d] = '%s' invalide. "
                        u"Attendu : %s" % (i, c, VALID_CONTINUITIES))
            self._continuities = list(continuities)

        self._name = name
        self._n_points = n_points
        self._sample_mode = sample_mode
        self._tolerance = tolerance
        self._cache = {}

        # Overrides per-segment (index -> valeur)
        self._alloc_overrides = {}   # segment_index -> n_points
        self._mode_overrides = {}    # segment_index -> sample_mode

        # Verifier les jonctions
        self._check_junctions()

    def _check_junctions(self):
        u"""Verifie la continuite aux jonctions (avertissement si violee)."""
        for i in range(len(self._segments) - 1):
            s_left = self._segments[i]
            s_right = self._segments[i + 1]
            cont = self._continuities[i]

            # C0 : position
            gap = np.linalg.norm(
                s_left.end_cpoint - s_right.start_cpoint)
            if gap > 1e-10:
                logger.warning(
                    u"Jonction %d : gap C0 = %.2e (attendu < 1e-10)"
                    % (i, gap))

            # C1 : direction tangente
            if cont in ('C1', 'C2'):
                tg_l = s_left.tangent(1.0)
                tg_r = s_right.tangent(0.0)
                cross = abs(tg_l[0] * tg_r[1] - tg_l[1] * tg_r[0])
                if cross > 1e-6:
                    logger.warning(
                        u"Jonction %d : ecart tangente C1 = %.2e"
                        % (i, cross))

            # C2 : courbure
            if cont == 'C2':
                k_l = s_left.curvature(1.0)
                k_r = s_right.curvature(0.0)
                if abs(k_l) + abs(k_r) > 1e-15:
                    k_err = abs(k_l - k_r) / max(
                        abs(k_l), abs(k_r), 1e-15)
                    if k_err > 1e-4:
                        logger.warning(
                            u"Jonction %d : ecart courbure C2 "
                            u"= %.2e" % (i, k_err))

    # ------------------------------------------------------------------
    #  Cache
    # ------------------------------------------------------------------

    def _invalidate(self, geometry=True):
        u"""Invalide le cache."""
        if geometry:
            self._cache.clear()
        else:
            for k in ('points', 'tangents', 'normals', 'curvatures'):
                self._cache.pop(k, None)

    # ------------------------------------------------------------------
    #  Properties de base
    # ------------------------------------------------------------------

    @property
    def segments(self):
        u"""Liste des segments Bezier (lecture seule)."""
        return list(self._segments)

    @property
    def n_segments(self):
        u"""Nombre de segments."""
        return len(self._segments)

    @property
    def continuities(self):
        u"""Niveaux de continuite aux jonctions."""
        return list(self._continuities)

    @property
    def name(self):
        u"""Nom de la spline."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def degree(self):
        u"""Degre(s) des segments.

        Retourne un int si tous les segments ont le meme degre,
        sinon une liste d'entiers.
        """
        degrees = [s.degree for s in self._segments]
        if len(set(degrees)) == 1:
            return degrees[0]
        return degrees

    @property
    def control_points(self):
        u"""Points de controle concatenes (sans doublons aux jonctions)."""
        parts = [self._segments[0].control_points]
        for s in self._segments[1:]:
            # Sauter le premier point (= dernier du segment precedent)
            parts.append(s.control_points[1:])
        return np.vstack(parts)

    @property
    def start_cpoint(self):
        u"""Premier point de controle."""
        return self._segments[0].start_cpoint

    @property
    def end_cpoint(self):
        u"""Dernier point de controle."""
        return self._segments[-1].end_cpoint

    # ------------------------------------------------------------------
    #  Echantillonnage
    # ------------------------------------------------------------------

    @property
    def n_points(self):
        u"""Nombre total de points echantillonnes."""
        return self._n_points

    @n_points.setter
    def n_points(self, value):
        if value < 2:
            raise ValueError(
                u"n_points doit etre >= 2, recu %d" % value)
        if value != self._n_points:
            self._n_points = value
            self._invalidate(geometry=False)

    @property
    def sample_mode(self):
        u"""Mode d'echantillonnage."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, value):
        if value != self._sample_mode:
            self._sample_mode = value
            self._invalidate(geometry=False)

    @property
    def tolerance(self):
        u"""Tolerance pour le mode adaptive."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        if value != self._tolerance:
            self._tolerance = value
            self._invalidate(geometry=False)

    # --- Overrides per-segment ---

    def segment_n_points(self, k):
        u"""Nombre de points effectif pour le segment k.

        Retourne l'override si defini, sinon l'allocation calculee.

        :param k: index du segment
        :type k: int
        :returns: nombre de points
        :rtype: int
        """
        if k in self._alloc_overrides:
            return self._alloc_overrides[k]
        return self._allocate_points(self._n_points)[k]

    def set_segment_n_points(self, k, n):
        u"""Definit un override de nombre de points pour le segment k.

        :param k: index du segment
        :type k: int
        :param n: nombre de points (>= 2)
        :type n: int
        """
        n = int(n)
        if n < 2:
            raise ValueError(
                u"n_points doit etre >= 2, recu %d" % n)
        if self._alloc_overrides.get(k) != n:
            self._alloc_overrides[k] = n
            self._invalidate(geometry=False)

    def clear_segment_n_points(self, k):
        u"""Supprime l'override de nombre de points pour le segment k."""
        if k in self._alloc_overrides:
            del self._alloc_overrides[k]
            self._invalidate(geometry=False)

    def segment_sample_mode(self, k):
        u"""Mode d'echantillonnage effectif pour le segment k.

        Retourne l'override si defini, sinon le mode global.

        :param k: index du segment
        :type k: int
        :returns: 'curvilinear' ou 'adaptive'
        :rtype: str
        """
        return self._mode_overrides.get(k, self._sample_mode)

    def set_segment_sample_mode(self, k, mode):
        u"""Definit un override de mode d'echantillonnage pour le segment k.

        :param k: index du segment
        :type k: int
        :param mode: 'curvilinear' ou 'adaptive'
        :type mode: str
        """
        if mode not in ('curvilinear', 'adaptive'):
            raise ValueError(
                u"Mode inconnu '%s'" % mode)
        if self._mode_overrides.get(k) != mode:
            self._mode_overrides[k] = mode
            self._invalidate(geometry=False)

    def set_segment_degree(self, k, target_degree):
        u"""Change le degre du segment k par elevation ou reduction.

        :param k: index du segment (0-indexed)
        :type k: int
        :param target_degree: degre cible (>= 1)
        :type target_degree: int
        :raises ValueError: si target_degree < 1
        :raises IndexError: si k hors limites
        """
        if k < 0 or k >= self.n_segments:
            raise IndexError(
                u"Index %d hors limites pour %d segments"
                % (k, self.n_segments))
        target_degree = int(target_degree)
        if target_degree < 1:
            raise ValueError(
                u"Le degre cible doit etre >= 1, recu %d"
                % target_degree)
        seg = self._segments[k]
        current = seg.degree
        if target_degree > current:
            seg.elevate(target_degree - current)
        elif target_degree < current:
            seg.reduce(current - target_degree)
        else:
            return
        self._invalidate(geometry=True)

    def clear_segment_overrides(self, k):
        u"""Supprime tous les overrides pour le segment k."""
        changed = False
        if k in self._alloc_overrides:
            del self._alloc_overrides[k]
            changed = True
        if k in self._mode_overrides:
            del self._mode_overrides[k]
            changed = True
        if changed:
            self._invalidate(geometry=False)

    @property
    def points(self):
        u"""Points echantillonnes sur la spline, ndarray(n_points, 2)."""
        if 'points' not in self._cache:
            self._cache['points'] = self.sample(
                self._n_points, self._sample_mode,
                self._tolerance)
        return self._cache['points'].copy()

    @property
    def tangents(self):
        u"""Tangentes unitaires aux points echantillonnes."""
        if 'tangents' not in self._cache:
            self._cache['tangents'] = self._sample_property(
                'tangent')
        return self._cache['tangents'].copy()

    @property
    def normals(self):
        u"""Normales unitaires aux points echantillonnes."""
        if 'normals' not in self._cache:
            self._cache['normals'] = self._sample_property(
                'normal')
        return self._cache['normals'].copy()

    @property
    def curvatures(self):
        u"""Courbures signees aux points echantillonnes."""
        if 'curvatures' not in self._cache:
            self._cache['curvatures'] = self._sample_property(
                'curvature')
        return self._cache['curvatures'].copy()

    def _sample_property(self, prop_name):
        u"""Echantillonne une propriete geometrique sur la spline."""
        alloc = self._allocate_points(self._n_points)
        parts = []
        for k, n_k in enumerate(alloc):
            seg = self._segments[k]
            t_vals = seg._sample_t_values(
                n_k, self._sample_mode, self._tolerance)
            if prop_name == 'tangent':
                vals = seg.tangent(t_vals)
            elif prop_name == 'normal':
                vals = seg.normal(t_vals)
            elif prop_name == 'curvature':
                vals = seg.curvature(t_vals)
            else:
                raise ValueError(prop_name)
            # Supprimer le premier point sauf pour le premier segment
            if k > 0:
                vals = vals[1:]
            parts.append(vals)
        if parts and parts[0].ndim == 1:
            return np.concatenate(parts)
        return np.vstack(parts)

    def sample(self, n, mode='curvilinear', tolerance=None):
        u"""Echantillonne la spline en n points.

        Les points sont repartis entre les segments au prorata de
        leur longueur d'arc. Les jonctions ne sont pas dupliquees.
        Les overrides per-segment (``_alloc_overrides``,
        ``_mode_overrides``) sont appliques le cas echeant.

        :param n: nombre de points
        :type n: int
        :param mode: 'curvilinear' ou 'adaptive'
        :type mode: str
        :param tolerance: tolerance pour le mode adaptive
        :type tolerance: float or None
        :returns: points echantillonnes, ndarray(m, 2)
        :rtype: numpy.ndarray
        """
        alloc = self._allocate_points(n)
        # Appliquer les overrides de nombre de points
        for k, nk in self._alloc_overrides.items():
            if 0 <= k < len(alloc):
                alloc[k] = nk
        parts = []
        for k, n_k in enumerate(alloc):
            seg_mode = self._mode_overrides.get(k, mode)
            seg_tol = tolerance
            # Tolerance par defaut pour le mode adaptive
            if seg_mode == 'adaptive' and seg_tol is None:
                seg_tol = 1e-3
            seg_pts = self._segments[k].sample(
                n_k, seg_mode, seg_tol)
            # Supprimer le premier point sauf pour le premier segment
            if k > 0:
                seg_pts = seg_pts[1:]
            parts.append(seg_pts)
        return np.vstack(parts)

    def _allocate_points(self, n):
        u"""Repartit n points entre les segments au prorata de l'arc.

        :returns: liste d'entiers, un par segment
        :rtype: list[int]
        """
        ns = self.n_segments
        if ns == 1:
            return [n]

        # Longueurs d'arc de chaque segment
        arc_lengths = []
        for seg in self._segments:
            t_fine, s_cum = seg._arc_length_table()
            arc_lengths.append(s_cum[-1])
        total = sum(arc_lengths)
        if total < 1e-15:
            # Degenere : repartition uniforme
            base = n // ns
            alloc = [base] * ns
            alloc[-1] += n - sum(alloc)
            return alloc

        # Points internes = n + (ns - 1) pour compenser les jonctions
        n_internal = n + ns - 1
        alloc = []
        for arc in arc_lengths:
            nk = max(2, int(round(arc / total * n_internal)))
            alloc.append(nk)

        # Ajuster pour que le total soit exact
        diff = n_internal - sum(alloc)
        # Ajouter/retrancher au plus grand segment
        idx_max = int(np.argmax(arc_lengths))
        alloc[idx_max] = max(2, alloc[idx_max] + diff)

        return alloc

    # ------------------------------------------------------------------
    #  Parametrisation globale
    # ------------------------------------------------------------------

    def _resolve_t(self, t):
        u"""Convertit un parametre global en (index_segment, t_local).

        :param t: parametre dans [0, N], scalaire
        :type t: float
        :returns: (k, t_local) avec k dans [0, N-1], t_local dans [0, 1]
        :rtype: tuple(int, float)
        """
        n = self.n_segments
        if t < 0 or t > n:
            raise ValueError(
                u"t=%.4f hors limites [0, %d]" % (t, n))
        if t == n:
            return n - 1, 1.0
        k = int(t)
        k = min(k, n - 1)
        return k, t - k

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, t):
        u"""Evalue la spline en t (parametre global dans [0, N]).

        :param t: scalaire ou array de parametres
        :returns: point(s) sur la courbe
        :rtype: numpy.ndarray
        """
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            k, tl = self._resolve_t(float(t))
            return self._segments[k].evaluate(tl)
        result = np.empty((len(t), 2), dtype=float)
        for i, ti in enumerate(t):
            k, tl = self._resolve_t(float(ti))
            result[i] = self._segments[k].evaluate(tl)
        return result

    def derivative(self, t, order=1):
        u"""Derivee en t (parametre global).

        :param t: scalaire ou array
        :param order: ordre de derivation (1 ou 2)
        :returns: vecteur(s) derivee
        """
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            k, tl = self._resolve_t(float(t))
            return self._segments[k].derivative(tl, order)
        result = np.empty((len(t), 2), dtype=float)
        for i, ti in enumerate(t):
            k, tl = self._resolve_t(float(ti))
            result[i] = self._segments[k].derivative(tl, order)
        return result

    def tangent(self, t):
        u"""Tangente unitaire en t (parametre global)."""
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            k, tl = self._resolve_t(float(t))
            return self._segments[k].tangent(tl)
        result = np.empty((len(t), 2), dtype=float)
        for i, ti in enumerate(t):
            k, tl = self._resolve_t(float(ti))
            result[i] = self._segments[k].tangent(tl)
        return result

    def normal(self, t):
        u"""Normale unitaire en t (parametre global)."""
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            k, tl = self._resolve_t(float(t))
            return self._segments[k].normal(tl)
        result = np.empty((len(t), 2), dtype=float)
        for i, ti in enumerate(t):
            k, tl = self._resolve_t(float(ti))
            result[i] = self._segments[k].normal(tl)
        return result

    def curvature(self, t):
        u"""Courbure signee en t (parametre global)."""
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            k, tl = self._resolve_t(float(t))
            return self._segments[k].curvature(tl)
        result = np.empty(len(t), dtype=float)
        for i, ti in enumerate(t):
            k, tl = self._resolve_t(float(ti))
            result[i] = self._segments[k].curvature(tl)
        return result

    # ------------------------------------------------------------------
    #  Projection / Split
    # ------------------------------------------------------------------

    def project(self, point):
        u"""Projette un point sur la spline (point le plus proche).

        Recherche grossiere par echantillonnage, puis raffinement Newton
        sur la distance ||S(t) - P||^2.

        :param point: coordonnees (x, y)
        :type point: array-like, shape (2,)
        :returns: (t_global, point_sur_courbe, distance)
        :rtype: tuple(float, numpy.ndarray, float)
        """
        point = np.asarray(point, dtype=float)
        N = self.n_segments
        n_fine = max(2000, N * 1000)
        t_fine = np.linspace(0, N, n_fine)
        pts_curve = self.evaluate(t_fine)

        # Distance au plus proche
        dx = pts_curve[:, 0] - point[0]
        dy = pts_curve[:, 1] - point[1]
        dists_sq = dx * dx + dy * dy
        i_min = int(np.argmin(dists_sq))
        t0 = float(t_fine[i_min])

        # Raffinement Newton (4 iterations)
        for _ in range(4):
            t0 = max(0.0, min(float(N), t0))
            k, tl = self._resolve_t(t0)
            seg = self._segments[k]
            bt = seg.evaluate(tl)
            d1 = seg.derivative(tl, order=1)
            d2 = seg.derivative(tl, order=2)
            diff = bt - point
            num = float(np.dot(diff, d1))
            den = float(np.dot(d1, d1) + np.dot(diff, d2))
            if abs(den) > 1e-15:
                tl -= num / den
            t0 = float(k) + tl
        t0 = max(0.0, min(float(N), t0))

        pt_closest = self.evaluate(t0)
        dist = float(np.linalg.norm(pt_closest - point))
        return t0, pt_closest, dist

    def split(self, t):
        u"""Scinde la spline au parametre global t.

        Le segment contenant t est remplace par 2 sous-segments
        via Bezier.split (De Casteljau exact, continuite C2).

        :param t: parametre global dans ]0, N[
        :type t: float
        :returns: nouvelle BezierSpline avec n_segments + 1 segments
        :rtype: BezierSpline
        :raises ValueError: si t hors limites ou sur une jonction
        """
        t = float(t)
        N = self.n_segments
        if t <= 0.0 or t >= N:
            raise ValueError(
                u"t doit etre dans ]0, %d[, recu %.6f" % (N, t))

        k, t_local = self._resolve_t(t)
        if t_local <= 0.0 or t_local >= 1.0:
            raise ValueError(
                u"t=%.6f tombe sur une jonction (t_local=%.6f)"
                % (t, t_local))

        # Scinder le segment k
        sub = self._segments[k].split(t_local)
        sub_segs = sub.segments  # 2 Bezier

        # Assembler les segments
        new_segments = (self._segments[:k]
                        + sub_segs
                        + self._segments[k + 1:])

        # Assembler les continuites
        new_continuities = (self._continuities[:k]
                            + ['C2']
                            + self._continuities[k:])

        new_spline = BezierSpline(
            new_segments,
            continuities=new_continuities,
            name=self._name,
            n_points=self._n_points,
            sample_mode=self._sample_mode,
            tolerance=self._tolerance)

        # Reporter les overrides (segment k supprime, segments > k decales)
        for idx, val in self._alloc_overrides.items():
            if idx < k:
                new_spline._alloc_overrides[idx] = val
            elif idx > k:
                new_spline._alloc_overrides[idx + 1] = val
        for idx, val in self._mode_overrides.items():
            if idx < k:
                new_spline._mode_overrides[idx] = val
            elif idx > k:
                new_spline._mode_overrides[idx + 1] = val

        return new_spline

    # ------------------------------------------------------------------
    #  Transformations (retournent self)
    # ------------------------------------------------------------------

    def translate(self, dx, dy):
        u"""Translation de tous les segments."""
        for seg in self._segments:
            seg.translate(dx, dy)
        self._invalidate(geometry=True)
        return self

    def rotate(self, angle_deg, center=None):
        u"""Rotation de tous les segments.

        :param angle_deg: angle en degres (+ = antihoraire)
        :param center: centre de rotation (defaut : premier point)
        """
        if center is None:
            center = self._segments[0].start_cpoint
        for seg in self._segments:
            seg.rotate(angle_deg, center=center)
        self._invalidate(geometry=True)
        return self

    def scale(self, factor, center=None):
        u"""Homothetie de tous les segments.

        :param factor: facteur d'echelle
        :param center: centre (defaut : premier point)
        """
        if center is None:
            center = self._segments[0].start_cpoint
        for seg in self._segments:
            seg.scale(factor, center=center)
        self._invalidate(geometry=True)
        return self

    def translate_cpoint(self, index, vector):
        u"""Translate un point de controle par son index global.

        L'index global correspond a la numerotation dans ``control_points``
        (sans doublons aux jonctions).  Si le point est a une jonction,
        les deux segments adjacents sont mis a jour.

        :param index: index global (supporte l'indexation negative)
        :type index: int
        :param vector: vecteur de translation (dx, dy)
        :type vector: array-like, shape (2,)
        :returns: self
        :rtype: BezierSpline
        """
        total = sum(s.degree for s in self._segments) + 1
        if index < -total or index >= total:
            raise IndexError(
                u"Index %d hors limites pour %d points de controle"
                % (index, total))
        if index < 0:
            index += total

        vector = np.asarray(vector, dtype=float)

        offset = 0
        for k, seg in enumerate(self._segments):
            d = seg.degree
            if index <= offset + d:
                local = index - offset
                seg.translate_cpoint(local, vector)
                # Jonction : mettre a jour le segment suivant aussi
                if local == d and k + 1 < len(self._segments):
                    self._segments[k + 1].translate_cpoint(0, vector)
                break
            offset += d

        self._invalidate(geometry=True)
        return self

    def reverse(self):
        u"""Inverse le sens de parcours de la spline.

        Inverse l'ordre des segments, retourne chaque segment,
        et inverse la liste des continuites.
        """
        self._segments = self._segments[::-1]
        for seg in self._segments:
            seg.reverse()
        self._continuities = self._continuities[::-1]
        self._invalidate(geometry=True)
        return self

    # ------------------------------------------------------------------
    #  Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return "BezierSpline('%s', %d segments, degres=%s)" % (
            self._name, self.n_segments,
            [s.degree for s in self._segments])
