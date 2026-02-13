#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Courbe de Bezier 2D de degre arbitraire.

Evaluation par l'algorithme de De Casteljau.
Derivees par reduction de degre sur les differences de points de controle.

Creation::

    # Cubique (4 points de controle)
    b = Bezier([[0,0], [100,200], [300,200], [400,0]])

    # Evaluation
    pt = b.evaluate(0.5)          # ndarray(2,)
    pts = b.evaluate([0, .25, .5, .75, 1])  # ndarray(5, 2)

    # Derivees, tangente, normale, courbure
    dp = b.derivative(0.5)        # B'(0.5)
    t  = b.tangent(0.5)           # vecteur unitaire
    n  = b.normal(0.5)            # perpendiculaire a t
    k  = b.curvature(0.5)         # courbure signee

@author: Nervures
@date: 2026-02
"""

import math

import numpy as np

# --------------------------------------------------------------------------
#  Algorithme de De Casteljau (fonction utilitaire)
# --------------------------------------------------------------------------

def _de_casteljau(points, t):
    u"""Evalue une courbe de Bezier en t par l'algorithme de De Casteljau.

    :param points: points de controle, ndarray(n+1, 2)
    :param t: parametre scalaire dans [0, 1]
    :returns: point sur la courbe, ndarray(2,)
    """
    pts = points.copy()
    n = len(pts) - 1
    for r in range(1, n + 1):
        pts[:n - r + 1] = (1.0 - t) * pts[:n - r + 1] + t * pts[1:n - r + 2]
    return pts[0]


# --------------------------------------------------------------------------
#  Classe Bezier
# --------------------------------------------------------------------------

class Bezier(object):
    u"""Courbe de Bezier 2D de degre arbitraire.

    Stockage des points de controle en coordonnees (x, y).
    Degre n = nombre de points de controle - 1.
    """

    def __init__(self, control_points, name='Sans nom',
                 n_points=100, sample_mode='curvilinear', tolerance=None,
                 degree=None, max_dev=None, smoothing=0.0,
                 start_tangent=None, end_tangent=None):
        u"""
        Si ``degree`` est None, ``control_points`` sont les points de
        controle P0..Pn (comportement standard).

        Si ``degree`` est un entier, ``control_points`` sont des points
        cibles et la courbe est ajustee par moindres carres au degre
        demande (appel a :meth:`approximate`).

        Si ``degree`` vaut ``'find'`` et ``max_dev`` est fourni,
        le degre minimal satisfaisant la deviation max est recherche
        automatiquement.

        :param control_points: points de controle ou points cibles
        :type control_points: numpy.ndarray or list
        :param name: nom de la courbe
        :type name: str
        :param n_points: nombre de points pour l'echantillonnage par defaut
        :type n_points: int
        :param sample_mode: mode d'echantillonnage ('curvilinear' ou 'adaptive')
        :type sample_mode: str
        :param tolerance: tolerance pour le mode adaptive (None = non defini)
        :type tolerance: float or None
        :param degree: si fourni, les points sont des cibles et la courbe
            est ajustee a ce degre par moindres carres. ``'find'`` pour
            recherche automatique du degre minimal.
        :type degree: int, str or None
        :param max_dev: deviation max autorisee (requis si degree='find')
        :type max_dev: float or None
        :param smoothing: poids de regularisation par differences secondes
            (0 = pas de lissage). Utilise seulement si ``degree`` est fourni.
        :type smoothing: float
        :param start_tangent: direction tangente en t=0 (angle deg ou vecteur)
        :type start_tangent: float, tuple or None
        :param end_tangent: direction tangente en t=1 (angle deg ou vecteur)
        :type end_tangent: float, tuple or None
        """
        pts = np.asarray(control_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                u"control_points doit etre un tableau (*, 2), "
                u"recu shape %s" % str(pts.shape))
        if len(pts) < 2:
            raise ValueError(
                u"Il faut au moins 2 points, recu %d" % len(pts))
        self._name = name
        self._n_points = n_points
        self._sample_mode = sample_mode
        self._tolerance = tolerance
        self._cache = {}
        if degree is not None:
            if degree == 'find':
                self._cpts = np.zeros((2, 2), dtype=float)
            else:
                self._cpts = np.zeros((int(degree) + 1, 2), dtype=float)
            self.approximate(pts, degree=degree, max_dev=max_dev,
                             smoothing=smoothing,
                             start_tangent=start_tangent,
                             end_tangent=end_tangent)
        else:
            self._cpts = pts

    # ------------------------------------------------------------------
    #  Cache
    # ------------------------------------------------------------------

    def _invalidate(self, geometry=True):
        u"""Invalide le cache interne.

        :param geometry: si True, la geometrie (points de controle) a change,
            tout le cache est efface. Si False, seul l'echantillonnage
            (n_points, sample_mode, tolerance) a change.
        :type geometry: bool
        """
        if geometry:
            self._cache.clear()
        else:
            self._cache.pop('points', None)
            self._cache.pop('tangents', None)
            self._cache.pop('normals', None)
            self._cache.pop('curvatures', None)

    # ------------------------------------------------------------------
    #  Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return "Bezier('%s', degre=%d, %d pts)" % (
            self._name, self.degree, len(self._cpts))

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def control_points(self):
        u"""Points de controle, ndarray(n+1, 2)."""
        return self._cpts

    @property
    def degree(self):
        u"""Degre de la courbe (n = nb_points - 1)."""
        return len(self._cpts) - 1

    @property
    def name(self):
        u"""Nom de la courbe."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def n_points(self):
        u"""Nombre de points pour l'echantillonnage par defaut."""
        return self._n_points

    @n_points.setter
    def n_points(self, value):
        if value < 2:
            raise ValueError(u"n_points doit etre >= 2, recu %d" % value)
        value = int(value)
        if value != self._n_points:
            self._n_points = value
            self._invalidate(geometry=False)

    @property
    def sample_mode(self):
        u"""Mode d'echantillonnage ('curvilinear' ou 'adaptive')."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, value):
        if value not in ('curvilinear', 'adaptive'):
            raise ValueError(
                u"Mode inconnu '%s'. Attendu : 'curvilinear', 'adaptive'"
                % value)
        if value != self._sample_mode:
            self._sample_mode = value
            self._invalidate(geometry=False)

    @property
    def tolerance(self):
        u"""Tolerance pour le mode adaptive (None = non defini)."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        if value != self._tolerance:
            self._tolerance = value
            self._invalidate(geometry=False)

    @property
    def points(self):
        u"""Points echantillonnes sur la courbe, ndarray(n_points, 2).

        Utilise les attributs n_points, sample_mode et tolerance.
        Le resultat est mis en cache et recalcule uniquement si les
        points de controle ou les parametres d'echantillonnage changent.
        """
        if 'points' in self._cache:
            return self._cache['points'].copy()
        result = self.sample(self._n_points, mode=self._sample_mode,
                             tolerance=self._tolerance)
        self._cache['points'] = result
        return result.copy()

    @property
    def tangents(self):
        u"""Tangentes unitaires aux points echantillonnes, ndarray(n_points, 2).

        Utilise les memes parametres que la property ``points``.
        Le resultat est mis en cache.
        """
        if 'tangents' in self._cache:
            return self._cache['tangents'].copy()
        result, _t = self.sample_tangents(
            self._n_points, self._sample_mode, self._tolerance)
        self._cache['tangents'] = result
        return result.copy()

    @property
    def normals(self):
        u"""Normales unitaires aux points echantillonnes, ndarray(n_points, 2).

        Utilise les memes parametres que la property ``points``.
        Le resultat est mis en cache.
        """
        if 'normals' in self._cache:
            return self._cache['normals'].copy()
        result, _t = self.sample_normals(
            self._n_points, self._sample_mode, self._tolerance)
        self._cache['normals'] = result
        return result.copy()

    @property
    def curvatures(self):
        u"""Courbures signees aux points echantillonnes, ndarray(n_points,).

        Utilise les memes parametres que la property ``points``.
        Le resultat est mis en cache.
        """
        if 'curvatures' in self._cache:
            return self._cache['curvatures'].copy()
        result, _t = self.sample_curvatures(
            self._n_points, self._sample_mode, self._tolerance)
        self._cache['curvatures'] = result
        return result.copy()

    def cpoint(self, index):
        u"""Retourne une copie du point de controle d'index donne.

        Supporte l'indexation negative (ex: -1 = dernier point).

        :param index: index du point de controle
        :type index: int
        :returns: coordonnees (x, y) du point
        :rtype: numpy.ndarray, shape (2,)
        """
        n = len(self._cpts)
        if index < -n or index >= n:
            raise IndexError(
                u"Index %d hors limites pour %d points de controle"
                % (index, n))
        return self._cpts[index].copy()

    def control_tangent(self, index):
        u"""Vecteur unitaire du segment de controle P[index] -> P[index+1].

        Supporte l'indexation negative (ex: -1 = avant-dernier -> dernier).

        :param index: index du segment (0 a n-1)
        :type index: int
        :returns: vecteur unitaire du segment
        :rtype: numpy.ndarray, shape (2,)
        """
        n_seg = len(self._cpts) - 1
        if index < -n_seg or index >= n_seg:
            raise IndexError(
                u"Index %d hors limites pour %d segments"
                % (index, n_seg))
        # Normaliser l'index negatif
        if index < 0:
            index += n_seg
        v = self._cpts[index + 1] - self._cpts[index]
        norm = np.linalg.norm(v)
        if norm < 1e-15:
            return np.zeros(2)
        return v / norm

    def translate_cpoint(self, index, vector):
        u"""Translate un point de controle selon un vecteur.

        Supporte l'indexation negative.

        :param index: index du point de controle
        :type index: int
        :param vector: vecteur de translation (dx, dy)
        :type vector: array-like, shape (2,)
        :returns: self (pour chainage)
        :rtype: Bezier
        """
        n = len(self._cpts)
        if index < -n or index >= n:
            raise IndexError(
                u"Index %d hors limites pour %d points de controle"
                % (index, n))
        self._cpts[index] += np.asarray(vector, dtype=float)
        self._invalidate(geometry=True)
        return self

    def translate(self, dx, dy):
        u"""Translation de la courbe.

        :param dx: deplacement en x
        :type dx: float
        :param dy: deplacement en y
        :type dy: float
        :returns: self (pour chainage)
        :rtype: Bezier
        """
        self._cpts[:, 0] += dx
        self._cpts[:, 1] += dy
        self._invalidate(geometry=True)
        return self

    def rotate(self, angle_deg, center=None):
        u"""Rotation de la courbe.

        :param angle_deg: angle en degres (positif = antihoraire)
        :type angle_deg: float
        :param center: centre de rotation (defaut : P0)
        :type center: array-like or None
        :returns: self (pour chainage)
        :rtype: Bezier
        """
        if center is None:
            center = self._cpts[0].copy()
        center = np.asarray(center, dtype=float)
        a = math.radians(angle_deg)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        pts = self._cpts - center
        x_new = pts[:, 0] * cos_a - pts[:, 1] * sin_a
        y_new = pts[:, 0] * sin_a + pts[:, 1] * cos_a
        self._cpts = np.column_stack([x_new, y_new]) + center
        self._invalidate(geometry=True)
        return self

    def scale(self, factor, center=None):
        u"""Homothetie de la courbe.

        :param factor: facteur d'echelle
        :type factor: float
        :param center: centre de l'homothetie (defaut : P0)
        :type center: array-like or None
        :returns: self (pour chainage)
        :rtype: Bezier
        """
        if center is None:
            center = self._cpts[0].copy()
        center = np.asarray(center, dtype=float)
        self._cpts = center + (self._cpts - center) * factor
        self._invalidate(geometry=True)
        return self

    def reverse(self):
        u"""Inverse le parametrage de la courbe.

        t=0 devient t=1 et vice-versa : P0 <-> Pn, P1 <-> Pn-1, etc.
        La geometrie est identique, seul le sens de parcours change.

        :returns: self (pour chainage)
        :rtype: Bezier
        """
        self._cpts = self._cpts[::-1].copy()
        self._invalidate(geometry=True)
        return self

    def elevate(self, times=1):
        u"""Eleve le degre de la courbe sans modifier sa forme.

        L'elevation de degre n vers n+1 produit n+2 points de controle
        Q0..Q(n+1) tels que :

        - Q0 = P0
        - Qi = (i/(n+1)) * P(i-1) + (1 - i/(n+1)) * Pi   pour i=1..n
        - Q(n+1) = Pn

        :param times: nombre d'elevations successives (defaut 1)
        :type times: int
        :returns: self (pour chainage)
        :rtype: Bezier
        """
        if times < 1:
            raise ValueError(
                u"times doit etre >= 1, recu %d" % times)
        for _ in range(times):
            n = self.degree
            P = self._cpts
            Q = np.empty((n + 2, 2), dtype=float)
            Q[0] = P[0]
            Q[n + 1] = P[n]
            for i in range(1, n + 1):
                alpha = float(i) / (n + 1)
                Q[i] = alpha * P[i - 1] + (1.0 - alpha) * P[i]
            self._cpts = Q
        self._invalidate(geometry=True)
        return self

    def reduce(self, times=1, clamp_ends=True, clamp_tangents=True):
        u"""Reduit le degre de la courbe (approximation).

        Contrairement a l'elevation (exacte), la reduction est une
        approximation. L'algorithme utilise un double balayage
        avant/arriere avec fusion ponderee.

        :param times: nombre de reductions successives (defaut 1)
        :type times: int
        :param clamp_ends: conserver les extremites P0 et Pn (defaut True)
        :type clamp_ends: bool
        :param clamp_tangents: conserver les tangentes aux extremites
            (defaut True). Implique clamp_ends=True.
        :type clamp_tangents: bool
        :returns: self (pour chainage)
        :rtype: Bezier
        """
        if times < 1:
            raise ValueError(
                u"times doit etre >= 1, recu %d" % times)
        if clamp_tangents:
            clamp_ends = True
        for _ in range(times):
            n = self.degree
            if n < 2:
                raise ValueError(
                    u"Impossible de reduire en dessous du degre 1 "
                    u"(degre actuel = %d)" % n)
            self._cpts = self._reduce_once(clamp_ends, clamp_tangents)
        self._invalidate(geometry=True)
        return self

    def _reduce_once(self, clamp_ends, clamp_tangents):
        u"""Reduction de degre n vers n-1 (une etape).

        Balayage avant : Q_f en partant de P0 (ou P0 + tangente).
        Balayage arriere : Q_b en partant de Pn (ou Pn - tangente).
        Les points libres sont un melange pondere des deux balayages.

        :returns: nouveaux points de controle, ndarray(n, 2)
        """
        n = self.degree
        m = n - 1  # degre cible
        P = self._cpts

        # --- Balayage avant ---
        Q_f = np.empty((m + 1, 2), dtype=float)
        Q_f[0] = P[0]
        if clamp_tangents and m >= 2:
            # B'(0) = n*(P1-P0) => Q1 = P0 + (n/m)*(P1-P0)
            Q_f[1] = P[0] + float(n) / m * (P[1] - P[0])
            fwd_start = 2
        else:
            fwd_start = 1
        for i in range(fwd_start, m + 1):
            alpha = float(i) / n
            Q_f[i] = (P[i] - alpha * Q_f[i - 1]) / (1.0 - alpha)

        # --- Balayage arriere ---
        Q_b = np.empty((m + 1, 2), dtype=float)
        Q_b[m] = P[n]
        if clamp_tangents and m >= 2:
            # B'(1) = n*(Pn-P(n-1)) => Q(m-1) = Pn - (n/m)*(Pn-P(n-1))
            Q_b[m - 1] = P[n] - float(n) / m * (P[n] - P[n - 1])
            bwd_start = m - 2
        else:
            bwd_start = m - 1
        for i in range(bwd_start, -1, -1):
            alpha = float(i + 1) / n
            Q_b[i] = (P[i + 1] - (1.0 - alpha) * Q_b[i + 1]) / alpha

        # --- Delimitation des zones fixes et libres ---
        if clamp_tangents and m >= 2:
            left_end = 2       # Q[0], Q[1] fixes par l'avant
            right_start = m - 1  # Q[m-1], Q[m] fixes par l'arriere
        elif clamp_ends:
            left_end = 1       # Q[0] fixe
            right_start = m    # Q[m] fixe
        else:
            left_end = 0
            right_start = m + 1

        Q = np.empty((m + 1, 2), dtype=float)

        # Points fixes cote gauche
        for i in range(min(left_end, m + 1)):
            Q[i] = Q_f[i]

        # Points fixes cote droit
        for i in range(max(right_start, 0), m + 1):
            Q[i] = Q_b[i]

        # Chevauchement (ex: cubique -> quadratique avec tangentes)
        for i in range(max(right_start, 0), min(left_end, m + 1)):
            Q[i] = 0.5 * (Q_f[i] + Q_b[i])

        # Points libres : fusion ponderee avant/arriere
        if left_end <= right_start - 1:
            for i in range(left_end, right_start):
                span = right_start - 1 - left_end
                if span == 0:
                    w = 0.5
                else:
                    w = float(i - left_end) / span
                Q[i] = (1.0 - w) * Q_f[i] + w * Q_b[i]

        return Q

    @staticmethod
    def _bernstein_matrix(n, t):
        u"""Matrice des polynomes de Bernstein.

        N[j, i] = B_{i,n}(t_j) = C(n,i) * t_j^i * (1 - t_j)^(n-i)

        :param n: degre de la courbe
        :type n: int
        :param t: valeurs du parametre, ndarray(m,)
        :type t: numpy.ndarray
        :returns: matrice de base, ndarray(m, n+1)
        :rtype: numpy.ndarray
        """
        m = len(t)
        N = np.zeros((m, n + 1))
        binom = np.ones(n + 1)
        for i in range(1, n + 1):
            binom[i] = binom[i - 1] * (n - i + 1) / i
        for i in range(n + 1):
            N[:, i] = binom[i] * t**i * (1.0 - t)**(n - i)
        return N

    @staticmethod
    def _smoothing_matrix(n, clamp_ends=True):
        u"""Matrice de regularisation par differences secondes.

        Construit la matrice D2 telle que D2 @ P donne les differences
        secondes du polygone de controle : P[i-1] - 2*P[i] + P[i+1].
        Si ``clamp_ends``, seuls les points interieurs sont penalises
        (lignes correspondant a i = 2 .. n-2).

        :param n: degre de la courbe (n+1 points de controle)
        :type n: int
        :param clamp_ends: exclure les extremites de la penalisation
        :type clamp_ends: bool
        :returns: matrice (k, n+1) avec k = n-1 ou n-3 selon clamp_ends
        :rtype: numpy.ndarray
        """
        # Differences secondes completes : (n-1) lignes
        rows = n - 1
        D2 = np.zeros((rows, n + 1))
        for i in range(rows):
            D2[i, i] = 1.0
            D2[i, i + 1] = -2.0
            D2[i, i + 2] = 1.0

        if clamp_ends and rows > 2:
            # Exclure la premiere et derniere ligne (touchent P0, Pn)
            D2 = D2[1:-1]

        return D2

    @staticmethod
    def _parse_tangent(tangent):
        u"""Convertit une specification de tangente en vecteur unitaire.

        :param tangent: None, angle en degres (float) ou vecteur (tuple/list)
        :returns: vecteur unitaire ndarray(2,) ou None
        """
        if tangent is None:
            return None
        if isinstance(tangent, (int, float)):
            a = math.radians(float(tangent))
            return np.array([math.cos(a), math.sin(a)])
        d = np.asarray(tangent, dtype=float).ravel()
        if len(d) != 2:
            raise ValueError(
                u"tangent doit etre 2D, recu %d composantes" % len(d))
        nrm = np.linalg.norm(d)
        if nrm < 1e-15:
            raise ValueError(u"tangent ne doit pas etre un vecteur nul")
        return d / nrm

    def _solve_clamped_with_tangents(self, N, points, n,
                                     d_start, d_end, smoothing):
        u"""Resout les moindres carres avec extremites fixees et tangentes.

        P0 = points[0], Pn = points[-1] (fixes).
        Si d_start: P1 = P0 + alpha * d_start  (1 DOF au lieu de 2).
        Si d_end:   Pn-1 = Pn - gamma * d_end  (1 DOF au lieu de 2).
        Les autres Pi restent libres (2 DOF chacun).

        Comme alpha et gamma couplent x et y, on empile les deux
        coordonnees dans un seul systeme.

        :param N: matrice de Bernstein (m+1, n+1)
        :param points: points cibles (m+1, 2)
        :param n: degre
        :param d_start: direction unitaire start ou None
        :param d_end: direction unitaire end ou None
        :param smoothing: poids de regularisation
        :returns: points de controle (n+1, 2)
        """
        m1 = len(points)
        P0, Pn = points[0], points[-1]
        has_ds = d_start is not None
        has_de = d_end is not None

        # Indices des points de controle libres (2 DOF chacun)
        i0 = 2 if has_ds else 1
        i1 = n - 2 if has_de else n - 1
        free_idx = list(range(i0, i1 + 1))
        n_free = len(free_idx)

        # Layout des inconnues : [alpha?, Q0x, Q0y, Q1x, Q1y, ..., gamma?]
        c_a = 0 if has_ds else -1
        c_f = 1 if has_ds else 0
        c_g = c_f + 2 * n_free if has_de else -1
        n_unk = c_f + 2 * n_free + (1 if has_de else 0)

        # RHS : soustraire les contributions connues (P0, Pn, et les
        # parties constantes de P1 = P0 + alpha*d et Pn-1 = Pn - gamma*d)
        rhs = points - np.outer(N[:, 0], P0) - np.outer(N[:, n], Pn)
        if has_ds:
            rhs -= np.outer(N[:, 1], P0)
        if has_de:
            rhs -= np.outer(N[:, n - 1], Pn)
        b = rhs.ravel()  # [x0, y0, x1, y1, ...]

        # --- Matrice de fitting (2*m1 lignes, n_unk colonnes) ---
        A = np.zeros((2 * m1, n_unk))
        if has_ds:
            A[0::2, c_a] = N[:, 1] * d_start[0]
            A[1::2, c_a] = N[:, 1] * d_start[1]
        for k, i in enumerate(free_idx):
            cx = c_f + 2 * k
            A[0::2, cx] = N[:, i]
            A[1::2, cx + 1] = N[:, i]
        if has_de:
            A[0::2, c_g] = -N[:, n - 1] * d_end[0]
            A[1::2, c_g] = -N[:, n - 1] * d_end[1]

        # --- Regularisation (differences secondes) ---
        if smoothing > 0 and n >= 3:
            free_col = {i: c_f + 2 * k for k, i in enumerate(free_idx)}
            n_sd = n - 1  # nombre de differences secondes
            A_s = np.zeros((2 * n_sd, n_unk))
            b_s = np.zeros(2 * n_sd)
            for di, i in enumerate(range(1, n)):
                rx, ry = 2 * di, 2 * di + 1
                for coeff, j in [(1.0, i - 1), (-2.0, i), (1.0, i + 1)]:
                    if j == 0:
                        b_s[rx] -= coeff * P0[0]
                        b_s[ry] -= coeff * P0[1]
                    elif j == n:
                        b_s[rx] -= coeff * Pn[0]
                        b_s[ry] -= coeff * Pn[1]
                    elif has_ds and j == 1:
                        A_s[rx, c_a] += coeff * d_start[0]
                        A_s[ry, c_a] += coeff * d_start[1]
                        b_s[rx] -= coeff * P0[0]
                        b_s[ry] -= coeff * P0[1]
                    elif has_de and j == n - 1:
                        A_s[rx, c_g] -= coeff * d_end[0]
                        A_s[ry, c_g] -= coeff * d_end[1]
                        b_s[rx] -= coeff * Pn[0]
                        b_s[ry] -= coeff * Pn[1]
                    elif j in free_col:
                        cx = free_col[j]
                        A_s[rx, cx] += coeff
                        A_s[ry, cx + 1] += coeff
            sl = np.sqrt(smoothing)
            A = np.vstack([A, sl * A_s])
            b = np.concatenate([b, sl * b_s])

        # --- Resolution ---
        u = np.linalg.lstsq(A, b, rcond=None)[0]

        # --- Reconstruction des points de controle ---
        cpts = np.empty((n + 1, 2))
        cpts[0] = P0
        cpts[n] = Pn
        if has_ds:
            cpts[1] = P0 + u[c_a] * d_start
        for k, i in enumerate(free_idx):
            cx = c_f + 2 * k
            cpts[i] = [u[cx], u[cx + 1]]
        if has_de:
            cpts[n - 1] = Pn - u[c_g] * d_end
        return cpts

    def approximate(self, points, degree=None, clamp_ends=True, max_iter=5,
                    max_dev=None, smoothing=0.0,
                    start_tangent=None, end_tangent=None):
        u"""Approche un ensemble de points par moindres carres regularises.

        Met a jour les points de controle pour approcher au mieux
        les points donnes. La parametrisation initiale est par longueur
        de corde, puis affinee par reprojection iterative.

        Le parametre ``smoothing`` controle la regularisation par
        differences secondes du polygone de controle. Plus la valeur
        est elevee, plus le polygone est lisse (mais moins fidele
        aux points cibles).

        ``start_tangent`` / ``end_tangent`` imposent la direction de la
        tangente aux extremites. Le point de controle P1 (resp. Pn-1)
        est contraint a se deplacer le long d'une droite passant par
        P0 (resp. Pn) dans la direction imposee (1 DOF au lieu de 2).

        :param points: points cibles, ndarray(m, 2) avec m >= degree + 1
        :type points: numpy.ndarray or list
        :param degree: degre cible (defaut : degre actuel), ou ``'find'``
        :type degree: int, str or None
        :param clamp_ends: matcher les extremites exactement (defaut True)
        :type clamp_ends: bool
        :param max_iter: iterations de reprojection (defaut 5)
        :type max_iter: int
        :param max_dev: deviation max autorisee (requis si degree='find')
        :type max_dev: float or None
        :param smoothing: poids de regularisation (0 = aucun lissage)
        :type smoothing: float
        :param start_tangent: direction tangente en t=0 : angle en degres
            (float) ou vecteur direction (tuple). None = libre.
        :type start_tangent: float, tuple or None
        :param end_tangent: direction tangente en t=1 : angle en degres
            (float) ou vecteur direction (tuple). None = libre.
        :type end_tangent: float, tuple or None
        :returns: self (pour chainage)
        :rtype: Bezier
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                u"points doit etre un tableau (m, 2), recu shape %s"
                % str(points.shape))
        smoothing = max(0.0, float(smoothing))
        d_start = self._parse_tangent(start_tangent)
        d_end = self._parse_tangent(end_tangent)
        has_tangents = d_start is not None or d_end is not None

        # --- Mode recherche automatique du degre ---
        if degree == 'find':
            if max_dev is None or max_dev <= 0:
                raise ValueError(
                    u"max_dev doit etre > 0 en mode degree='find', "
                    u"recu %s" % str(max_dev))
            m = len(points) - 1
            max_degree = m
            # Degre minimum selon les contraintes de tangente
            min_degree = 1
            if d_start is not None and d_end is not None:
                min_degree = 3
            elif has_tangents:
                min_degree = 2
            for n_try in range(min_degree, max_degree + 1):
                self.approximate(points, degree=n_try,
                                 clamp_ends=clamp_ends, max_iter=max_iter,
                                 smoothing=smoothing,
                                 start_tangent=start_tangent,
                                 end_tangent=end_tangent)
                dev, _t = self.max_deviation(points)
                if dev <= max_dev:
                    return self
            return self

        if degree is None:
            degree = self.degree
        n = int(degree)
        if n < 1:
            raise ValueError(u"degree doit etre >= 1, recu %d" % n)
        m = len(points) - 1
        if m < n:
            raise ValueError(
                u"Il faut au moins %d points pour un degre %d, recu %d"
                % (n + 1, n, m + 1))
        max_iter = max(0, int(max_iter))

        # Desactiver les tangentes si le degre est insuffisant
        if n < 2:
            d_start = d_end = None
            has_tangents = False
        elif n < 3 and d_start is not None and d_end is not None:
            d_end = None  # garder seulement la contrainte start

        # --- Matrice de lissage (utilisee seulement sans tangentes) ---
        use_smoothing = smoothing > 0.0 and n >= 3
        if use_smoothing and not has_tangents:
            D2_full = self._smoothing_matrix(n, clamp_ends=clamp_ends)

        # --- Parametrisation initiale par longueur de corde ---
        diffs = np.diff(points, axis=0)
        chord = np.sqrt(np.sum(diffs**2, axis=1))
        total = np.sum(chord)

        if total < 1e-15:
            self._cpts = np.tile(points[0], (n + 1, 1))
            self._invalidate(geometry=True)
            return self

        t = np.zeros(m + 1)
        t[1:] = np.cumsum(chord) / total
        t[-1] = 1.0

        # --- Boucle fit + reprojection ---
        for iteration in range(max_iter + 1):
            N = self._bernstein_matrix(n, t)

            if clamp_ends and has_tangents:
                # Systeme empile x/y avec contraintes de tangente
                cpts = self._solve_clamped_with_tangents(
                    N, points, n, d_start, d_end, smoothing)
            elif clamp_ends:
                if n == 1:
                    cpts = np.array([points[0], points[-1]], dtype=float)
                elif n == 2 or not use_smoothing:
                    rhs = (points
                           - np.outer(N[:, 0], points[0])
                           - np.outer(N[:, n], points[-1]))
                    N_inner = N[:, 1:n]
                    P_inner = np.linalg.lstsq(
                        N_inner, rhs, rcond=None)[0]
                    cpts = np.empty((n + 1, 2), dtype=float)
                    cpts[0] = points[0]
                    cpts[-1] = points[-1]
                    cpts[1:-1] = P_inner
                else:
                    rhs = (points
                           - np.outer(N[:, 0], points[0])
                           - np.outer(N[:, n], points[-1]))
                    N_inner = N[:, 1:n]
                    D2_inner = D2_full[:, 1:n]
                    lam = smoothing
                    A = np.vstack([N_inner, np.sqrt(lam) * D2_inner])
                    b = np.vstack(
                        [rhs, np.zeros((D2_inner.shape[0], 2))])
                    P_inner = np.linalg.lstsq(A, b, rcond=None)[0]
                    cpts = np.empty((n + 1, 2), dtype=float)
                    cpts[0] = points[0]
                    cpts[-1] = points[-1]
                    cpts[1:-1] = P_inner
            else:
                if not use_smoothing:
                    cpts = np.linalg.lstsq(N, points, rcond=None)[0]
                else:
                    D2 = self._smoothing_matrix(n, clamp_ends=False)
                    lam = smoothing
                    A = np.vstack([N, np.sqrt(lam) * D2])
                    b = np.vstack(
                        [points, np.zeros((D2.shape[0], 2))])
                    cpts = np.linalg.lstsq(A, b, rcond=None)[0]

            self._cpts = cpts

            if iteration == max_iter or m <= 1:
                break

            # --- Reprojection : meilleur t pour chaque point ---
            n_fine = min(5000, max(1000, 5 * (m + 1)))
            t_fine = np.linspace(0, 1, n_fine)
            pts_curve = self.evaluate(t_fine)

            inner_pts = points[1:m]
            if len(inner_pts) == 0:
                break

            dx = inner_pts[:, None, 0] - pts_curve[None, :, 0]
            dy = inner_pts[:, None, 1] - pts_curve[None, :, 1]
            dists_sq = dx**2 + dy**2
            idx = np.argmin(dists_sq, axis=1)
            t0 = t_fine[idx].copy()

            for _ in range(4):
                bt = self.evaluate(t0)
                d1 = self.derivative(t0, order=1)
                d2 = self.derivative(t0, order=2)
                diff = bt - inner_pts
                num = np.sum(diff * d1, axis=1)
                den = (np.sum(d1 * d1, axis=1)
                       + np.sum(diff * d2, axis=1))
                mask = np.abs(den) > 1e-15
                t0[mask] -= num[mask] / den[mask]
                t0 = np.clip(t0, 0.0, 1.0)

            t_new = np.empty(m + 1)
            t_new[0] = 0.0
            t_new[1:m] = t0
            t_new[m] = 1.0
            for j in range(1, m + 1):
                if t_new[j] <= t_new[j - 1]:
                    t_new[j] = t_new[j - 1] + 1e-10

            if np.max(np.abs(t_new - t)) < 1e-8:
                break
            t = t_new

        self._invalidate(geometry=True)
        return self

    def max_deviation(self, points):
        u"""Deviation max entre des points et la courbe.

        Pour chaque point, calcule la distance au point le plus proche
        sur la courbe (projection sur un echantillonnage fin + raffinement
        Newton), et retourne le maximum ainsi que le parametre t
        correspondant.

        :param points: points a tester, ndarray(m, 2)
        :type points: numpy.ndarray or list
        :returns: (deviation_max, t_max) — deviation maximale et
            parametre t ou elle se produit
        :rtype: tuple(float, float)
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                u"points doit etre un tableau (m, 2), recu shape %s"
                % str(points.shape))
        m = len(points)
        n_fine = min(5000, max(1000, 10 * m))
        t_fine = np.linspace(0, 1, n_fine)
        pts_curve = self.evaluate(t_fine)

        # Distance au plus proche sur la courbe (vectorise)
        dx = points[:, None, 0] - pts_curve[None, :, 0]
        dy = points[:, None, 1] - pts_curve[None, :, 1]
        dists_sq = dx**2 + dy**2
        idx = np.argmin(dists_sq, axis=1)
        t0 = t_fine[idx].copy()

        # Raffinement Newton (minimise ||B(t) - P||^2)
        for _ in range(4):
            bt = self.evaluate(t0)
            d1 = self.derivative(t0, order=1)
            d2 = self.derivative(t0, order=2)
            diff = bt - points
            num = np.sum(diff * d1, axis=1)
            den = np.sum(d1 * d1, axis=1) + np.sum(diff * d2, axis=1)
            mask = np.abs(den) > 1e-15
            t0[mask] -= num[mask] / den[mask]
            t0 = np.clip(t0, 0.0, 1.0)

        # Distance finale apres raffinement
        bt = self.evaluate(t0)
        dists = np.sqrt(np.sum((bt - points)**2, axis=1))
        i_max = np.argmax(dists)
        return float(dists[i_max]), float(t0[i_max])

    @property
    def start_cpoint(self):
        u"""Premier point de controle P0, ndarray(2,)."""
        return self._cpts[0].copy()

    @property
    def end_cpoint(self):
        u"""Dernier point de controle Pn, ndarray(2,)."""
        return self._cpts[-1].copy()

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, t):
        u"""Evalue la courbe en t (De Casteljau).

        :param t: parametre dans [0, 1], scalaire ou array
        :type t: float or numpy.ndarray
        :returns: point(s) sur la courbe
        :rtype: ndarray(2,) si t scalaire, ndarray(m, 2) si t array
        """
        t = np.asarray(t, dtype=float)
        if t.ndim == 0:
            return _de_casteljau(self._cpts, float(t))
        result = np.empty((len(t), 2), dtype=float)
        for i, ti in enumerate(t):
            result[i] = _de_casteljau(self._cpts, float(ti))
        return result

    # ------------------------------------------------------------------
    #  Derivees
    # ------------------------------------------------------------------

    def derivative(self, t, order=1):
        u"""Derivee d'ordre 1 ou 2 en t.

        Derivee d'ordre 1 : B'(t) = n * Bezier(delta_P)(t)
          ou delta_P[i] = P[i+1] - P[i]

        Derivee d'ordre 2 : B''(t) = n*(n-1) * Bezier(delta2_P)(t)
          ou delta2_P[i] = P[i+2] - 2*P[i+1] + P[i]

        :param t: parametre dans [0, 1], scalaire ou array
        :param order: ordre de derivation (1 ou 2)
        :type order: int
        :returns: vecteur(s) derivee
        :rtype: ndarray
        """
        if order not in (1, 2):
            raise ValueError(
                u"Ordre de derivation %d non supporte (1 ou 2)" % order)

        n = self.degree
        if order == 1:
            if n < 1:
                t = np.asarray(t, dtype=float)
                if t.ndim == 0:
                    return np.zeros(2)
                return np.zeros((len(t), 2))
            # Points de controle de la derivee : n * (P[i+1] - P[i])
            delta = n * (self._cpts[1:] - self._cpts[:-1])
            if len(delta) == 1:
                # Degre 1 : derivee constante
                t = np.asarray(t, dtype=float)
                if t.ndim == 0:
                    return delta[0].copy()
                return np.tile(delta[0], (len(t), 1))
            deriv = Bezier(delta)
            return deriv.evaluate(t)

        else:  # order == 2
            if n < 2:
                t = np.asarray(t, dtype=float)
                if t.ndim == 0:
                    return np.zeros(2)
                return np.zeros((len(t), 2))
            # Differences secondes : n*(n-1) * (P[i+2] - 2*P[i+1] + P[i])
            delta2 = n * (n - 1) * (
                self._cpts[2:] - 2 * self._cpts[1:-1] + self._cpts[:-2])
            if len(delta2) == 1:
                # Degre 2 : derivee seconde constante
                t = np.asarray(t, dtype=float)
                if t.ndim == 0:
                    return delta2[0].copy()
                return np.tile(delta2[0], (len(t), 1))
            deriv2 = Bezier(delta2)
            return deriv2.evaluate(t)

    def tangent(self, t):
        u"""Vecteur tangent unitaire en t.

        :param t: parametre dans [0, 1], scalaire ou array
        :returns: vecteur(s) tangent(s) unitaire(s)
        :rtype: ndarray
        """
        d = self.derivative(t, order=1)
        if d.ndim == 1:
            norm = np.linalg.norm(d)
            if norm < 1e-15:
                return np.zeros(2)
            return d / norm
        else:
            norms = np.linalg.norm(d, axis=1, keepdims=True)
            norms = np.where(norms < 1e-15, 1.0, norms)
            return d / norms

    def normal(self, t):
        u"""Vecteur normal unitaire en t (rotation +90 deg de la tangente).

        :param t: parametre dans [0, 1], scalaire ou array
        :returns: vecteur(s) normal(aux) unitaire(s)
        :rtype: ndarray
        """
        tg = self.tangent(t)
        if tg.ndim == 1:
            return np.array([-tg[1], tg[0]])
        else:
            return np.column_stack([-tg[:, 1], tg[:, 0]])

    def curvature(self, t):
        u"""Courbure signee en t.

        kappa = (x' * y'' - y' * x'') / (x'^2 + y'^2)^(3/2)

        :param t: parametre dans [0, 1], scalaire ou array
        :returns: courbure(s)
        :rtype: float ou ndarray
        """
        d1 = self.derivative(t, order=1)
        d2 = self.derivative(t, order=2)

        if d1.ndim == 1:
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            norm_sq = d1[0]**2 + d1[1]**2
            denom = norm_sq ** 1.5
            if denom < 1e-30:
                return 0.0
            return float(cross / denom)
        else:
            cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
            norm_sq = d1[:, 0]**2 + d1[:, 1]**2
            denom = norm_sq ** 1.5
            denom = np.where(denom < 1e-30, 1.0, denom)
            return cross / denom

    # ------------------------------------------------------------------
    #  Echantillonnage
    # ------------------------------------------------------------------

    def _arc_length_table(self, n_fine=1000):
        u"""Calcule la table d'abscisse curviligne sur une grille fine.

        Le resultat est mis en cache (invalide quand la geometrie change).

        :param n_fine: nombre de points de la grille fine
        :returns: (t_fine, s_cum) — parametres et abscisse curviligne cumulee
        :rtype: tuple(ndarray, ndarray)
        """
        key = ('arc_length', n_fine)
        if key in self._cache:
            return self._cache[key]
        t_fine = np.linspace(0, 1, n_fine)
        pts = self.evaluate(t_fine)
        ds = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        s = np.zeros(n_fine)
        s[1:] = np.cumsum(ds)
        self._cache[key] = (t_fine, s)
        return t_fine, s

    def _sample_t_values(self, n, mode='curvilinear', tolerance=None):
        u"""Calcule les valeurs du parametre t pour un echantillonnage.

        Meme logique que :meth:`sample` mais retourne les parametres t
        au lieu des points evalues.

        :param n: nombre de points
        :type n: int
        :param mode: 'curvilinear' ou 'adaptive'
        :type mode: str
        :param tolerance: ecart max courbe-polyligne (requis si adaptive)
        :type tolerance: float or None
        :returns: parametres t, ndarray(n,)
        :rtype: numpy.ndarray
        """
        if n < 2:
            raise ValueError(u"n doit etre >= 2, recu %d" % n)

        n_fine = max(1000, 10 * n)
        t_fine, s = self._arc_length_table(n_fine)
        s_total = s[-1]

        if s_total < 1e-15:
            # Courbe degeneree
            return np.linspace(0, 1, n)

        if mode == 'curvilinear':
            s_targets = np.linspace(0, s_total, n)
            return np.interp(s_targets, s, t_fine)

        elif mode == 'adaptive':
            if tolerance is None:
                raise ValueError(
                    u"tolerance requise pour le mode 'adaptive'")
            if tolerance <= 0:
                raise ValueError(
                    u"tolerance doit etre > 0, recu %g" % tolerance)

            # Courbure absolue sur la grille fine
            kappa = np.abs(self.curvature(t_fine))

            # Lissage gaussien pour variation C1 de l'espacement
            sigma = max(1, n_fine // 50)
            kappa_smooth = self._gaussian_smooth(kappa, sigma)

            # Densite a partir de l'erreur cordale :
            #   erreur ~ (1/8) * kappa * ds^2
            #   ds_max = sqrt(8 * tol / kappa)
            #   densite = 1 / ds_max = sqrt(kappa / (8 * tol))
            density = np.sqrt(
                np.maximum(kappa_smooth, 1e-10) / (8.0 * tolerance))

            # Plancher : densite minimale pour ne pas avoir de trous
            # dans les zones rectilignes
            min_density = np.mean(density) * 0.1
            density = np.maximum(density, min_density)

            # Integrale de la densite le long de l'abscisse curviligne
            ds = np.diff(s)
            cum = np.zeros(n_fine)
            cum[1:] = np.cumsum(0.5 * (density[:-1] + density[1:]) * ds)

            # Repartir n points uniformement dans l'espace cumule
            targets = np.linspace(0, cum[-1], n)
            return np.interp(targets, cum, t_fine)

        else:
            raise ValueError(
                u"Mode inconnu '%s'. Attendu : 'curvilinear', 'adaptive'"
                % mode)

    def sample(self, n, mode='curvilinear', tolerance=None):
        u"""Echantillonne la courbe en n points.

        Deux modes disponibles :

        - ``'curvilinear'`` : points repartis uniformement en abscisse
          curviligne.

        - ``'adaptive'`` : points repartis selon la courbure locale pour
          limiter l'ecart entre la polyligne et la courbe theorique.
          La densite est proportionnelle a sqrt(kappa), lissee par
          convolution gaussienne pour assurer une variation C1 de
          l'espacement. Le parametre ``tolerance`` controle la
          distribution : plus la tolerance est petite, plus les zones
          de forte courbure concentrent de points.

        :param n: nombre de points
        :type n: int
        :param mode: 'curvilinear' ou 'adaptive'
        :type mode: str
        :param tolerance: ecart max courbe-polyligne (requis si adaptive)
        :type tolerance: float or None
        :returns: points echantillonnes, ndarray(n, 2)
        :rtype: numpy.ndarray
        """
        t = self._sample_t_values(n, mode, tolerance)
        return self.evaluate(t)

    def sample_tangents(self, n, mode='curvilinear', tolerance=None):
        u"""Calcule les tangentes unitaires aux points echantillonnes.

        :param n: nombre de points
        :type n: int
        :param mode: 'curvilinear' ou 'adaptive'
        :type mode: str
        :param tolerance: ecart max courbe-polyligne (requis si adaptive)
        :type tolerance: float or None
        :returns: (tangentes, t) — tangentes unitaires ndarray(n, 2)
            et parametres ndarray(n,)
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """
        t = self._sample_t_values(n, mode, tolerance)
        return self.tangent(t), t

    def sample_normals(self, n, mode='curvilinear', tolerance=None):
        u"""Calcule les normales unitaires aux points echantillonnes.

        :param n: nombre de points
        :type n: int
        :param mode: 'curvilinear' ou 'adaptive'
        :type mode: str
        :param tolerance: ecart max courbe-polyligne (requis si adaptive)
        :type tolerance: float or None
        :returns: (normales, t) — normales unitaires ndarray(n, 2)
            et parametres ndarray(n,)
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """
        t = self._sample_t_values(n, mode, tolerance)
        return self.normal(t), t

    def sample_curvatures(self, n, mode='curvilinear', tolerance=None):
        u"""Calcule la courbure aux points echantillonnes.

        :param n: nombre de points
        :type n: int
        :param mode: 'curvilinear' ou 'adaptive'
        :type mode: str
        :param tolerance: ecart max courbe-polyligne (requis si adaptive)
        :type tolerance: float or None
        :returns: (courbures, t) — courbures signees ndarray(n,)
            et parametres ndarray(n,)
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """
        t = self._sample_t_values(n, mode, tolerance)
        return self.curvature(t), t

    @staticmethod
    def _gaussian_smooth(values, sigma):
        u"""Lissage gaussien 1D (numpy pur).

        :param values: signal a lisser, ndarray(n,)
        :param sigma: ecart-type du noyau en nombre d'indices
        :returns: signal lisse, ndarray(n,)
        """
        radius = int(4 * sigma)
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma)**2)
        kernel /= kernel.sum()

        # Padding par extension des bords
        padded = np.empty(len(values) + 2 * radius)
        padded[:radius] = values[0]
        padded[radius:radius + len(values)] = values
        padded[radius + len(values):] = values[-1]

        smoothed = np.convolve(padded, kernel, mode='same')
        return smoothed[radius:radius + len(values)]

    # ------------------------------------------------------------------
    #  I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, filepath, **kwargs):
        u"""Charge des points depuis un fichier texte et cree une Bezier.

        Le fichier doit contenir deux colonnes (x y) separees par des
        espaces, tabulations, virgules ou points-virgules.
        Les lignes vides et celles commencant par '#' sont ignorees.
        La premiere ligne non-commentaire est utilisee comme nom si elle
        ne contient pas de valeurs numeriques.

        Les points lus sont traites comme des **points de controle**.
        Pour les utiliser comme points cibles, passer ``degree=<int>``
        dans les kwargs.

        :param filepath: chemin du fichier
        :type filepath: str
        :param kwargs: arguments supplementaires passes au constructeur
        :returns: instance Bezier
        :rtype: Bezier
        """
        import os
        filepath = str(filepath)
        if not os.path.isfile(filepath):
            raise IOError(u"Fichier introuvable : %s" % filepath)

        name = kwargs.pop('name', None)
        points = []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Remplacer les separateurs courants
                line = line.replace(';', ' ').replace(',', ' ').replace('\t', ' ')
                parts = line.split()
                try:
                    vals = [float(v) for v in parts[:2]]
                except ValueError:
                    # Ligne non numerique : utiliser comme nom
                    if name is None and not points:
                        name = line.strip()
                    continue
                if len(vals) == 2:
                    points.append(vals)

        if len(points) < 2:
            raise ValueError(
                u"Pas assez de points dans '%s' (%d trouves)"
                % (filepath, len(points)))

        if name is None:
            name = os.path.splitext(os.path.basename(filepath))[0]

        return cls(np.array(points), name=name, **kwargs)

    # ------------------------------------------------------------------
    #  Visualisation
    # ------------------------------------------------------------------

    def plot(self, ax=None, show=True, control_polygon=True,
             sample_points=True):
        u"""Trace la courbe de Bezier.

        :param ax: axes matplotlib existants (None = creation)
        :param show: appeler plt.show() a la fin
        :param control_polygon: afficher le polygone de controle
        :param sample_points: afficher les points echantillonnes (property points)
        :returns: axes matplotlib
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Courbe evaluee (haute resolution)
        t_vals = np.linspace(0, 1, 200)
        pts = self.evaluate(t_vals)
        ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=1.5, label=self._name)

        # Points echantillonnes
        if sample_points:
            sp = self.points
            ax.plot(sp[:, 0], sp[:, 1], 'b|', markersize=5,
                    label=u'%d points (%s)' % (self._n_points,
                                               self._sample_mode))

        if control_polygon:
            # Polygone de controle
            ax.plot(self._cpts[:, 0], self._cpts[:, 1],
                    'o--', color='gray', linewidth=0.8, markersize=4,
                    label=u'Polygone de controle')
            # P0 et Pn
            ax.plot(self._cpts[0, 0], self._cpts[0, 1],
                    'ro', markersize=6, label='P0')
            ax.plot(self._cpts[-1, 0], self._cpts[-1, 1],
                    'gs', markersize=6, label='P%d' % self.degree)

        ax.set_aspect('equal')
        ax.set_title(self._name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        if show:
            plt.show()

        return ax

def test_bezier_profil():
    b = Bezier.from_file('C:/Liclipse/workspace/AifoilTools/naca0012_normalized_intrados.dat', degree=16, max_dev=5, sample_mode='adaptive', tolerance=15.0)
    b.plot()
    return

if __name__ == "__main__":
    
    # Exemple d'utilisation
    # b = Bezier([[0, 0], [150, 200], [250, 200], [400, 0]], name='Exemple Cubique')
    # print(b)
    # print("points de controle =", b.control_points)
    # print(b.control_points[-1])  # dernier point de controle
    # print(b.cpoint(1))  # point de controle d'index 1
    # print(b.control_tangent(-1)) # tangent du dernier segment de controle
    # print("start =", b.start_cpoint)
    # print("end =", b.end_cpoint)
    # print("points shape =", b.points.shape)
    # print("P(0.5) =", b.evaluate(0.5))
    # print("B'(0.5) =", b.derivative(0.5))
    # print("Tangent(0.5) =", b.tangent(0.5))
    # print("Normal(0.5) =", b.normal(0.5))
    # print("Curvature(0.5) =", b.curvature(0.5))
    # b.sample_mode = 'adaptive'
    # b.tolerance = 5.0
    # b.translate_cpoint(1, [-20, -10])
    # b.plot()

    # b.elevate(times=2)
    # b.plot()
    # print("curvatures shape =", b.curvatures.shape)
    # print("curvatures =", b.curvatures)

    # b.rotate(45)
    # b.plot()

    
    
    

    # b = Bezier([[0, 0], [150, 200], [250, 200], [400, 0]], name='Exemple Cubique')
    # pts_b = b.points
    # print("Echantillonage adaptatif points =", pts_b)
    # b.plot()
    # c=Bezier([[0, 0], [150, 0], [250, 0], [400, 0]], name='approx')
    # c.approximate(pts_b, degree='find', clamp_ends=True, max_iter=10, max_dev=1.0)
    # pts_c=c.points
    
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # ax.plot(pts_b[:, 0], pts_b[:, 1], 'o', markersize=5, label='Points cibles')
    # ax.plot(pts_c[:, 0], pts_c[:, 1], 'r-', linewidth=2, label='Courbe approximee')
    # plt.show()
    # print("Deviation max =", c.max_deviation(pts_b))
    # c.plot()    

    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # ax.plot(pts_b[:, 0], pts_b[:, 1], 'o', markersize=5, label='Points cibles')
    # ax.plot(pts_c[:, 0], pts_c[:, 1], 'r-', linewidth=2, label='Courbe approximee')
    # plt.show()


    b = test_bezier_profil()
    