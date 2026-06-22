#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Braquage de volet (flap) sur un profil aerodynamique.

Construit l'articulation puis applique le braquage selon la geometrie
decrite dans ``flap_xfoil.pptx`` :

- Le centre d'articulation ``cf`` est le centre du cercle tangent a
  l'extrados et a l'intrados dont l'abscisse vaut ``Xf`` (en pourcent
  de corde).
- Chaque surface est coupee au point de tangence puis sa partie arriere
  (volet) est tournee de l'angle de braquage autour de ``cf``.
- Selon le signe du braquage et le cote, un **arc de cercle** comble le
  jeu (gap) ou une **intersection** trime le recouvrement.

Convention : braquage ``delta`` en degres, positif = bord de fuite vers
le HAUT, negatif = bord de fuite vers le BAS.

Regle par surface :

==========  =================  ====================
Cote        delta < 0 (bas)    delta > 0 (haut)
==========  =================  ====================
extrados    arc (gap)          intersection
intrados    intersection       arc (gap)
==========  =================  ====================

@author: Nervures
@date: 2026-06
"""

import os
import logging

import numpy as np

if __name__ == '__main__' and not __package__:
    import sys as _sys
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from model.profil_spline import ProfilSpline
else:
    from .profil_spline import ProfilSpline

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
#  Primitives geometriques
# ----------------------------------------------------------------------

def _dist_point_polyline(P, pts):
    u"""Distance mini de ``P`` a la polyligne + projete le plus proche.

    :param P: point (2,)
    :param pts: polyligne (n, 2)
    :returns: (distance, point_projete, index_segment, t_local)
    :rtype: tuple
    """
    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    ap = P - a
    denom = np.sum(ab * ab, axis=1)
    denom[denom == 0] = 1e-12
    t = np.clip(np.sum(ap * ab, axis=1) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.hypot(proj[:, 0] - P[0], proj[:, 1] - P[1])
    k = int(np.argmin(d))
    return d[k], proj[k], k, t[k]


def hinge_circle(extrados, intrados, xf):
    u"""Cercle tangent a l'extrados et a l'intrados, centre a ``x=xf``.

    Recherche par dichotomie l'ordonnee du centre telle que les distances
    aux deux surfaces soient egales (cercle inscrit touchant le haut et
    le bas).

    :param extrados: points BA->BF (x croissant), ndarray(n, 2)
    :param intrados: points BA->BF (x croissant), ndarray(m, 2)
    :param xf: abscisse du centre (mm)
    :type xf: float
    :returns: (centre C, rayon r, point ef, point if)
    :rtype: tuple(numpy.ndarray, float, numpy.ndarray, numpy.ndarray)
    """
    y_ext = float(np.interp(xf, extrados[:, 0], extrados[:, 1]))
    y_int = float(np.interp(xf, intrados[:, 0], intrados[:, 1]))
    lo, hi = y_int, y_ext  # intrados (bas) -> extrados (haut)
    for _ in range(80):
        yc = 0.5 * (lo + hi)
        C = np.array([xf, yc])
        de, _, _, _ = _dist_point_polyline(C, extrados)
        di, _, _, _ = _dist_point_polyline(C, intrados)
        if de > di:        # trop loin de l'extrados -> monter
            lo = yc
        else:
            hi = yc
    C = np.array([xf, 0.5 * (lo + hi)])
    de, ef, _, _ = _dist_point_polyline(C, extrados)
    di, if_, _, _ = _dist_point_polyline(C, intrados)
    r = 0.5 * (de + di)
    return C, r, ef, if_


def _rotate(pts, center, deg):
    u"""Rotation de points autour d'un centre (deg, sens trigo positif)."""
    a = np.radians(deg)
    ca, sa = np.cos(a), np.sin(a)
    d = np.asarray(pts, dtype=float) - center
    x = d[..., 0] * ca - d[..., 1] * sa
    y = d[..., 0] * sa + d[..., 1] * ca
    return np.stack([x, y], axis=-1) + center


def _seg_inter(p1, p2, p3, p4):
    u"""Intersection des segments [p1,p2] et [p3,p4], ou None."""
    r = p2 - p1
    s = p4 - p3
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-12:
        return None
    qp = p3 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return p1 + t * r, t, u
    return None


def _polyline_intersections(A, B):
    u"""Liste des croisements entre 2 polylignes : (P, iA, iB)."""
    out = []
    for i in range(len(A) - 1):
        for j in range(len(B) - 1):
            res = _seg_inter(A[i], A[i + 1], B[j], B[j + 1])
            if res is not None:
                out.append((res[0], i, j))
    return out


def _arc_points(C, r, a0_deg, a1_deg, n=40):
    u"""Points d'un arc de cercle de a0 a a1 (deg)."""
    ang = np.radians(np.linspace(a0_deg, a1_deg, n))
    return np.column_stack([C[0] + r * np.cos(ang), C[1] + r * np.sin(ang)])


def _surface_mode(side, delta):
    u"""Retourne 'arc' ou 'inter' selon le cote et le signe du braquage."""
    if side == 'extrados':
        return 'arc' if delta < 0 else 'inter'
    return 'inter' if delta < 0 else 'arc'


def _deflect_surface(surface, C, r, tang_pt, delta, mode):
    u"""Applique le braquage a une surface (extrados ou intrados).

    :param surface: points BA->BF (x croissant)
    :param C: centre d'articulation
    :param r: rayon du cercle
    :param tang_pt: point de tangence (ef ou if)
    :param delta: braquage (deg)
    :param mode: 'arc' (gap) ou 'inter' (recouvrement)
    :returns: courbe resultante BA->BF, ndarray(k, 2)
    :rtype: numpy.ndarray
    """
    _, _, k, _ = _dist_point_polyline(tang_pt, surface)
    avant = np.vstack([surface[:k + 1], tang_pt])
    arriere = np.vstack([tang_pt, surface[k + 1:]])
    arriere_rot = _rotate(arriere, C, delta)

    if mode == 'arc':
        a0 = np.degrees(np.arctan2(tang_pt[1] - C[1], tang_pt[0] - C[0]))
        arc = _arc_points(C, r, a0, a0 + delta)
        return np.vstack([avant, arc, arriere_rot])

    # mode == 'inter' : recouvrement -> trim a l'intersection
    inters = _polyline_intersections(surface, arriere_rot)
    if not inters:
        return np.vstack([avant, arriere_rot])
    P, iA, iB = min(inters, key=lambda it: np.hypot(*(it[0] - C)))
    avant_trim = np.vstack([surface[:iA + 1], P])
    arriere_trim = np.vstack([P, arriere_rot[iB + 1:]])
    return np.vstack([avant_trim, arriere_trim])


# ----------------------------------------------------------------------
#  API publique
# ----------------------------------------------------------------------

def apply_flap(profil, xf_percent, delta_deg, name=None):
    u"""Construit le profil braque (volet) a partir d'un profil.

    :param profil: profil source (mode discret ou spline)
    :type profil: ProfilSpline
    :param xf_percent: position de l'axe d'articulation, en pourcent de
        corde mesure depuis le bord d'attaque
    :type xf_percent: float
    :param delta_deg: braquage en degres (+ = bord de fuite vers le haut)
    :type delta_deg: float
    :param name: nom du profil resultant (defaut : derive du source)
    :type name: str or None
    :returns: profil braque (points discrets, non editable)
    :rtype: ProfilSpline
    """
    ext = np.asarray(profil.extrados, dtype=float)    # BA -> BF
    intr = np.asarray(profil.intrados, dtype=float)   # BA -> BF
    chord = profil.chord

    if name is None:
        name = u"%s (flap %+.1f° @ %.0f%%)" % (
            profil.name, delta_deg, xf_percent)

    # Braquage nul : le profil est inchange
    if abs(delta_deg) < 1e-9:
        return ProfilSpline(profil.points.copy(), name=name)

    # Abscisse de l'axe (corde mesuree depuis le BA)
    x_le = float(ext[0, 0])
    xf = x_le + (xf_percent / 100.0) * chord
    x_min = max(ext[0, 0], intr[0, 0])
    x_max = min(ext[-1, 0], intr[-1, 0])
    xf = float(np.clip(xf, x_min + 1e-6, x_max - 1e-6))

    C, r, ef, if_ = hinge_circle(ext, intr, xf)

    ext_r = _deflect_surface(
        ext, C, r, ef, delta_deg, _surface_mode('extrados', delta_deg))
    int_r = _deflect_surface(
        intr, C, r, if_, delta_deg, _surface_mode('intrados', delta_deg))

    # Reconstruction convention Selig : BF -> extrados -> BA -> intrados -> BF
    points = np.vstack([ext_r[::-1], int_r[1:]])
    return ProfilSpline(points, name=name)
