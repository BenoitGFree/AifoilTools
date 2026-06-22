#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Prototype geometrique + schema explicatif du braquage de volet (flap).

Construit l'articulation (cercle tangent extrados/intrados, centre a x=Xf)
puis applique le braquage selon les deux cas (arc / intersection) decrits
dans flap_xfoil.pptx. Genere une figure annotee.

Convention : braquage delta en degres, positif = bord de fuite vers le
HAUT, negatif = bord de fuite vers le BAS (rotation autour de cf).

Regle par surface :
  - extrados : delta<0 -> ARC (gap) ; delta>0 -> INTERSECTION (recouvrement)
  - intrados : delta<0 -> INTERSECTION ; delta>0 -> ARC

Usage : env_py3\\Scripts\\python.exe tools\\flap_schema.py
"""

import os
import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'sources'))

from model.profil_spline import ProfilSpline


# ----------------------------------------------------------------------
#  Primitives geometriques
# ----------------------------------------------------------------------

def _dist_point_polyline(P, pts):
    u"""Distance mini de P a la polyligne + point projete le plus proche.

    :returns: (dist, point_proche, index_segment, t_local)
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
    u"""Cercle tangent a l'extrados et a l'intrados, centre a x=xf.

    :param extrados: points BA->BF (x croissant)
    :param intrados: points BA->BF (x croissant)
    :param xf: abscisse du centre (mm)
    :returns: (centre C, rayon r, point ef, point if)
    """
    y_ext = np.interp(xf, extrados[:, 0], extrados[:, 1])
    y_int = np.interp(xf, intrados[:, 0], intrados[:, 1])
    lo, hi = y_int, y_ext  # intrados (bas) -> extrados (haut)
    C = np.array([xf, 0.5 * (lo + hi)])
    for _ in range(80):
        yc = 0.5 * (lo + hi)
        C = np.array([xf, yc])
        de, _, _, _ = _dist_point_polyline(C, extrados)
        di, _, _, _ = _dist_point_polyline(C, intrados)
        if de > di:        # trop loin de l'extrados -> monter
            lo = yc
        else:
            hi = yc
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


def _split_at_point(pts, k, P):
    u"""Coupe une polyligne au point P situe sur le segment k.

    :returns: (avant, arriere) avec P inclus dans les deux.
    """
    avant = np.vstack([pts[:k + 1], P])
    arriere = np.vstack([P, pts[k + 1:]])
    return avant, arriere


def _seg_inter(p1, p2, p3, p4):
    u"""Intersection de segments [p1,p2] et [p3,p4]. None si pas de croisement."""
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
    u"""Tous les croisements entre 2 polylignes : (P, iA, iB)."""
    out = []
    for i in range(len(A) - 1):
        for j in range(len(B) - 1):
            res = _seg_inter(A[i], A[i + 1], B[j], B[j + 1])
            if res is not None:
                P, _, _ = res
                out.append((P, i, j))
    return out


def _arc_points(C, r, a0_deg, a1_deg, n=40):
    u"""Points d'un arc de cercle de a0 a a1 (deg)."""
    ang = np.radians(np.linspace(a0_deg, a1_deg, n))
    return np.column_stack([C[0] + r * np.cos(ang), C[1] + r * np.sin(ang)])


# ----------------------------------------------------------------------
#  Braquage d'une surface
# ----------------------------------------------------------------------

def deflect_surface(surface, C, r, tang_pt, delta, mode):
    u"""Applique le braquage a une surface (extrados ou intrados).

    :param surface: points BA->BF (x croissant)
    :param C: centre d'articulation
    :param r: rayon du cercle
    :param tang_pt: point de tangence (ef ou if)
    :param delta: braquage (deg, +=haut)
    :param mode: 'arc' (gap) ou 'inter' (recouvrement)
    :returns: dict avec les morceaux pour le trace et la courbe resultante
    """
    _, _, k, _ = _dist_point_polyline(tang_pt, surface)
    avant, arriere = _split_at_point(surface, k, tang_pt)

    if mode == 'arc':
        # 1) partie avant inchangee (jusqu'a la tangence)
        # 2) arc de cercle d'angle delta
        # 3) partie arriere tournee de delta autour de C
        arriere_rot = _rotate(arriere, C, delta)
        a0 = np.degrees(np.arctan2(tang_pt[1] - C[1], tang_pt[0] - C[0]))
        arc = _arc_points(C, r, a0, a0 + delta)
        result = np.vstack([avant, arc, arriere_rot])
        return {'mode': 'arc', 'avant': avant, 'arriere_rot': arriere_rot,
                'arc': arc, 'result': result}

    # mode == 'inter' : recouvrement -> trim a l'intersection
    arriere_rot = _rotate(arriere, C, delta)
    inters = _polyline_intersections(surface, arriere_rot)
    if not inters:
        # pas de croisement (braquage faible) : on raccorde tel quel
        result = np.vstack([avant, arriere_rot])
        return {'mode': 'inter', 'avant': avant, 'arriere_rot': arriere_rot,
                'P': None, 'result': result}
    # intersection la plus proche du centre C
    P, iA, iB = min(inters, key=lambda it: np.hypot(*(it[0] - C)))
    avant_trim = np.vstack([surface[:iA + 1], P])
    arriere_trim = np.vstack([P, arriere_rot[iB + 1:]])
    result = np.vstack([avant_trim, arriere_trim])
    return {'mode': 'inter', 'avant': avant_trim,
            'arriere_rot': arriere_rot, 'P': P, 'result': result}


def surface_mode(side, delta):
    u"""Retourne 'arc' ou 'inter' selon le cote et le signe du braquage."""
    if delta == 0:
        return None
    if side == 'extrados':
        return 'arc' if delta < 0 else 'inter'
    else:  # intrados
        return 'inter' if delta < 0 else 'arc'


# ----------------------------------------------------------------------
#  Schema
# ----------------------------------------------------------------------

def main():
    p = ProfilSpline.from_naca('2412', n_points=300)
    p.normalize()
    ext = p.extrados   # BA -> BF
    intr = p.intrados
    chord = p.chord
    xf = 0.70 * chord
    C, r, ef, if_ = hinge_circle(ext, intr, xf)

    BLUE = '#1f77b4'
    RED = '#d62728'
    GREEN = '#2ca02c'
    GREY = '#888888'

    fig = Figure(figsize=(13, 9), dpi=110)
    FigureCanvasAgg(fig)

    # ---- (a) Construction de l'articulation ----
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(ext[:, 0], ext[:, 1], '-', color=BLUE, lw=1.5)
    ax.plot(intr[:, 0], intr[:, 1], '-', color=BLUE, lw=1.5)
    ax.add_patch(Circle(C, r, fill=False, edgecolor=GREEN, lw=1.8))
    ax.plot([C[0]], [C[1]], 'o', color=GREEN, ms=5)
    ax.plot([ef[0]], [ef[1]], 'o', color=GREEN, ms=5)
    ax.plot([if_[0]], [if_[1]], 'o', color=GREEN, ms=5)
    ax.annotate('cf', C, textcoords='offset points', xytext=(6, 4),
                color=GREEN, fontsize=11)
    ax.annotate('ef', ef, textcoords='offset points', xytext=(6, 4),
                color=GREEN, fontsize=11)
    ax.annotate('if', if_, textcoords='offset points', xytext=(6, -14),
                color=GREEN, fontsize=11)
    ax.plot([C[0], ef[0]], [C[1], ef[1]], '-', color=GREEN, lw=0.9)
    ax.plot([C[0], if_[0]], [C[1], if_[1]], '-', color=GREEN, lw=0.9)
    ax.axvline(xf, color=GREY, ls='--', lw=0.8)
    ax.annotate('x = Xf = %.0f%%' % (100 * xf / chord),
                (xf, C[1] + r + 12), color=GREY, fontsize=9,
                ha='center')
    ax.set_title(u"(a) Articulation : cercle tangent (centre cf a x=Xf)",
                 fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(xf - 220, chord + 30)
    ax.grid(True, alpha=0.3)

    # ---- (b) Extrados, braquage negatif -> ARC (gap) ----
    delta_neg = -15.0
    res = deflect_surface(ext, C, r, ef, delta_neg, 'arc')
    ax = fig.add_subplot(2, 2, 2)
    ax.add_patch(Circle(C, r, fill=False, edgecolor=GREEN, lw=1.0,
                        alpha=0.5))
    ax.plot(ext[:, 0], ext[:, 1], '-', color=GREY, lw=1.0, alpha=0.6,
            label='extrados d\'origine')
    ax.plot(res['arriere_rot'][:, 0], res['arriere_rot'][:, 1], '--',
            color=RED, lw=1.0, alpha=0.7, label=u'arriere tournee (delta)')
    ax.plot(res['result'][:, 0], res['result'][:, 1], '-', color=GREEN,
            lw=2.0, label=u'resultat')
    ax.plot(res['arc'][:, 0], res['arc'][:, 1], '-', color='#ff7f0e',
            lw=3.0, label='arc inserre')
    ax.plot([C[0]], [C[1]], 'o', color=GREEN, ms=4)
    ax.plot([ef[0]], [ef[1]], 'o', color='k', ms=4)
    ax.set_title(u"(b) Extrados, braquage NEGATIF (BdF bas) -> ARC",
                 fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(xf - 120, chord + 30)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ---- (c) Extrados, braquage positif -> INTERSECTION ----
    delta_pos = 15.0
    res = deflect_surface(ext, C, r, ef, delta_pos, 'inter')
    ax = fig.add_subplot(2, 2, 3)
    ax.add_patch(Circle(C, r, fill=False, edgecolor=GREEN, lw=1.0,
                        alpha=0.5))
    ax.plot(ext[:, 0], ext[:, 1], '-', color=GREY, lw=1.0, alpha=0.6,
            label='extrados d\'origine')
    ax.plot(res['arriere_rot'][:, 0], res['arriere_rot'][:, 1], '--',
            color=RED, lw=1.0, alpha=0.7, label=u'arriere tournee (delta)')
    ax.plot(res['result'][:, 0], res['result'][:, 1], '-', color=GREEN,
            lw=2.0, label=u'resultat')
    if res['P'] is not None:
        ax.plot([res['P'][0]], [res['P'][1]], 'o', color='#ff7f0e',
                ms=8, label='intersection')
    ax.plot([C[0]], [C[1]], 'o', color=GREEN, ms=4)
    ax.set_title(u"(c) Extrados, braquage POSITIF (BdF haut) -> INTERSECTION",
                 fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(xf - 120, chord + 60)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ---- (d) Profil complet braque (vert) ----
    delta = -15.0
    res_e = deflect_surface(ext, C, r, ef, delta,
                            surface_mode('extrados', delta))
    res_i = deflect_surface(intr, C, r, if_, delta,
                            surface_mode('intrados', delta))
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(ext[:, 0], ext[:, 1], '-', color=GREY, lw=1.0, alpha=0.5)
    ax.plot(intr[:, 0], intr[:, 1], '-', color=GREY, lw=1.0, alpha=0.5,
            label='profil d\'origine')
    ax.add_patch(Circle(C, r, fill=False, edgecolor=GREEN, lw=0.8,
                        alpha=0.4))
    ax.plot(res_e['result'][:, 0], res_e['result'][:, 1], '-',
            color=GREEN, lw=2.0)
    ax.plot(res_i['result'][:, 0], res_i['result'][:, 1], '-',
            color=GREEN, lw=2.0, label=u'profil braque (delta=%g deg)' % delta)
    ax.plot([C[0]], [C[1]], 'o', color=GREEN, ms=4)
    ax.set_title(u"(d) Profil complet braque (vert)", fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(-30, chord + 60)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        u"Braquage de volet (flap) - construction geometrique",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(_HERE, 'flap_schema.png')
    fig.savefig(out, dpi=110)
    print('Schema ecrit :', out)


if __name__ == '__main__':
    main()
