#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Test standalone foil2d avec execution reelle de XFoil.

Cree un profil NACA 2412 tel qu'Axile le fournira (via ProfilNormalise),
puis lance un calcul XFoil complet et affiche les resultats.

Usage::

    python sources/model/aerodynamique/foil2d/test_foil2d_xfoil.py

@author: Nervures
@date: 2026-02
"""

import os
import sys
import tempfile
import logging

# --- Setup des chemins pour les imports ---
_here = os.path.dirname(os.path.abspath(__file__))
_pkg = os.path.normpath(os.path.join(_here, '..', 'airfoiltools'))
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)

# --- Logging visible en console ---
logging.basicConfig(
    level=logging.INFO,
    format='%(name)-20s %(levelname)-8s %(message)s'
)

import numpy as np


# =========================================================================
# Creation du profil "a la Axile"
# =========================================================================

def get_profile_from_axile():
    u"""Cree un profil NACA 2412 comme Axile le ferait.

    Tente d'utiliser le modele Axile (ProfilNormalise).
    En fallback, genere les points NACA directement.

    :returns: (points, nom) - ndarray(n,2) et nom du profil
    :rtype: tuple(numpy.ndarray, str)
    """
    try:
        from model.profil.naca import naca4
        points = naca4('2412', n=80)
        print("Profil cree via model.profil.naca.naca4")
        print("  -> %d points, shape %s" % (points.shape[0], points.shape))
        print("  -> BF depart  : (%.4f, %.4f)" % (points[0, 0], points[0, 1]))
        print("  -> BA (milieu): (%.4f, %.4f)" % (points[79, 0], points[79, 1]))
        print("  -> BF fin     : (%.4f, %.4f)" % (points[-1, 0], points[-1, 1]))
        return points, 'NACA2412'

    except ImportError:
        print("Import Axile indisponible, generation NACA analytique")
        return _naca2412_standalone(), 'NACA2412'


def _naca2412_standalone(n=80):
    u"""Generation NACA 2412 autonome (sans dependance Axile).

    Parametres NACA 2412 : cambrure max 2%, position 40%, epaisseur 12%.
    """
    m, p, t = 0.02, 0.4, 0.12
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))  # demi-cosinus

    # Epaisseur
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x
                     - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    # Ligne de cambrure et derivee
    yc = np.where(x < p,
                  m / p**2 * (2 * p * x - x**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(x < p,
                   2 * m / p**2 * (p - x),
                   2 * m / (1 - p)**2 * (p - x))
    theta = np.arctan(dyc)

    # Extrados / intrados
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Assemblage : BF -> extrados -> BA -> intrados -> BF
    x_all = np.concatenate([xu[::-1], xl[1:]])
    y_all = np.concatenate([yu[::-1], yl[1:]])
    return np.column_stack([x_all, y_all])


# =========================================================================
# Execution du pipeline
# =========================================================================

def run_xfoil_analysis():
    u"""Lance une analyse XFoil complete sur un profil NACA 2412."""

    print("=" * 60)
    print("Test foil2d - Execution XFoil reelle")
    print("=" * 60)

    # 1. Creer le profil comme Axile le fournira
    print("\n--- 1. Creation du profil ---")
    profile_points, profile_name = get_profile_from_axile()

    # 2. Parametres de simulation
    params = {
        'RE': 500000,
        'RE_LIST': [300000, 500000, 1000000],
        'MACH': 0.0,
        'ALPHA_MIN': -5.0,
        'ALPHA_MAX': 12.0,
        'ALPHA_STEP': 0.5,
        'VISCOUS': True,
        'NCRIT': 9,
        'XTR_TOP': 0.01,
        'XTR_BOT': 0.01,
        'ITER': 200,
        'REPANEL': True,
        'NPANEL': 200,
        'TIMEOUT': 60,
    }

    # 3. Creer et lancer le pipeline
    print("\n--- 2. Lancement du pipeline ---")
    work_dir = tempfile.mkdtemp(prefix='foil2d_xfoil_')
    print("  Repertoire de travail : %s" % work_dir)

    from pipeline import FoilAnalysisPipeline
    pipeline = FoilAnalysisPipeline(
        solver='xfoil',
        work_dir=work_dir
    )

    results = pipeline.run(profile_points, params)

    # 4. Afficher les resultats
    print("\n--- 3. Resultats ---")
    display_results(results)

    # 5. Afficher le contenu du repertoire de travail
    print("\n--- 4. Fichiers generes ---")
    for f in sorted(os.listdir(work_dir)):
        size = os.path.getsize(os.path.join(work_dir, f))
        print("  %-35s %8d octets" % (f, size))

    print("\n  Repertoire : %s" % work_dir)
    print("  (non supprime pour inspection)")

    return results, work_dir


def display_results(results):
    u"""Affiche un resume des resultats."""

    # Warnings
    warnings = results.get('warnings', [])
    if warnings:
        print("\n  WARNINGS (%d) :" % len(warnings))
        for w in warnings:
            print("    - %s" % w)

    # Polaires
    polars = results.get('polars', {})
    if not polars:
        print("\n  AUCUNE POLAIRE — XFoil n'a probablement pas converge.")
        return

    for re_val in sorted(polars.keys()):
        polar = polars[re_val]
        alpha = polar['alpha']
        cl = polar['CL']
        cd = polar['CD']
        cm = polar['CM']

        print("\n  === Polaire Re = %g ===" % re_val)
        print("  %d points converges (alpha %.1f a %.1f)"
              % (len(alpha), alpha[0], alpha[-1]))

        # Tableau resume
        print("  %8s %8s %9s %8s %8s" % ('alpha', 'CL', 'CD', 'CM', 'CL/CD'))
        print("  %8s %8s %9s %8s %8s" % ('-----', '-----', '------',
                                           '-----', '------'))
        # Afficher un point sur 4
        step = max(1, len(alpha) // 8)
        for i in range(0, len(alpha), step):
            finesse = cl[i] / cd[i] if cd[i] > 1e-6 else float('inf')
            print("  %8.2f %8.4f %9.6f %8.4f %8.1f"
                  % (alpha[i], cl[i], cd[i], cm[i], finesse))

        # Grandeurs remarquables
        if len(cl) > 0:
            i_clmax = np.argmax(cl)
            finesse_array = np.where(cd > 1e-6, cl / cd, 0)
            i_fmax = np.argmax(finesse_array)
            print("\n  CL max     = %.4f  a alpha = %.1f deg"
                  % (cl[i_clmax], alpha[i_clmax]))
            print("  Finesse max = %.1f    a alpha = %.1f deg (CL=%.3f, CD=%.6f)"
                  % (finesse_array[i_fmax], alpha[i_fmax],
                     cl[i_fmax], cd[i_fmax]))

    # Cp
    cp_data = results.get('cp', {})
    n_cp = sum(len(v) for v in cp_data.values())
    print("\n  Distributions Cp : %d fichiers parses" % n_cp)

    # Couche limite
    bl_data = results.get('bl', {})
    print("  Couches limites  : %d fichiers parses" % len(bl_data))


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    try:
        results, work_dir = run_xfoil_analysis()
        print("\n" + "=" * 60)
        print("TEST TERMINE AVEC SUCCES")
        print("=" * 60)
    except Exception as e:
        print("\nERREUR : %s" % str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
