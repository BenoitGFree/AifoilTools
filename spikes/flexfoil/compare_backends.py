#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""Parite des backends via l'API Simulation : 'xfoil' vs 'flexfoil'.

Lance le MEME profil avec les MEMES parametres a travers les deux
backends derriere l'abstraction commune (Simulation/pipeline), puis
compare le contrat de donnees produit : polaires, Cp visqueux,
Cp non visqueux et couche limite. Valide que FlexFoil est un
remplacant transparent (transition forcee comprise, defaut XTR=0.2).

Doit tourner sous Python 3.11 (env_py3 du projet) avec flexfoil et
xfoil.exe disponibles.

    env_py3/Scripts/python.exe spikes/flexfoil/compare_backends.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'sources'))

import numpy as np  # noqa: E402

from model.profil import Profil          # noqa: E402
from model.simulation import Simulation  # noqa: E402

PARAMS = {
    'RE': 1.0e6,
    'ALPHA_MIN': -4.0, 'ALPHA_MAX': 10.0, 'ALPHA_STEP': 2.0,
    # XTR_TOP/XTR_BOT et NCRIT viennent des defauts (0.2 / 0.2 / 9),
    # identiques pour les deux backends -> comparaison a iso-parametres.
}
RE = 1.0e6
A_CP = 6.0    # incidence pour comparer Cp et couche limite


def run(solver):
    prof = Profil.from_naca('2412')
    sim = Simulation(prof, params=dict(PARAMS), solver=solver)
    sim.run()
    return sim.results


def interp_on(xa, ya, xb):
    u"""Interpole ya(xa) aux abscisses xb (tables triees par x)."""
    o = np.argsort(xa)
    return np.interp(xb, xa[o], ya[o])


def upper_surface(x, val):
    u"""Extrait l'extrados d'un contour Selig (sans sillage x>1).

    Coupe au bord d'attaque (x minimal) et garde l'arc allant du BF au
    BA, trie par x croissant. Indispensable pour comparer Cp ou Dstar
    surface a surface (sur un contour Selig chaque x apparait deux fois,
    une fois par face).
    """
    x = np.asarray(x, dtype=float)
    val = np.asarray(val, dtype=float)
    keep = x <= 1.0 + 1e-6
    x, val = x[keep], val[keep]
    le = int(np.argmin(x))
    xe, ve = x[:le + 1], val[:le + 1]
    o = np.argsort(xe)
    return xe[o], ve[o]


def compare_surface(name, xa, va, xb, vb, lo=0.05, hi=0.95):
    u"""Compare deux distributions extrados sur la zone [lo, hi]."""
    xe_a, ve_a = upper_surface(xa, va)
    xe_b, ve_b = upper_surface(xb, vb)
    m = (xe_b > lo) & (xe_b < hi)
    ref = np.interp(xe_b[m], xe_a, ve_a)
    err = np.abs(ve_b[m] - ref)
    print('  %-16s err max=%.4f moy=%.4f' % (name, err.max(), err.mean()))
    return err.max()


def main():
    print('=' * 70)
    print('PARITE BACKENDS via Simulation  -  NACA 2412, Re=1e6, XTR=0.2')
    print('=' * 70)

    xf = run('xfoil')
    ff = run('flexfoil')

    # --- Polaires ---
    px = xf.get_polar(RE)
    pf = ff.get_polar(RE)
    print('\n--- Polaire (alpha communs) ---')
    print('%6s %8s %8s %9s %8s %8s %9s %8s %8s' %
          ('alpha', 'CL_xf', 'CL_ff', 'dCL', 'CD_xf', 'CD_ff', 'dCD',
           'CM_xf', 'CM_ff'))
    common = sorted(set(np.round(px['alpha'], 3)) &
                    set(np.round(pf['alpha'], 3)))
    mdcl = mdcd = mdcm = 0.0
    for a in common:
        ix = np.argmin(np.abs(px['alpha'] - a))
        iff = np.argmin(np.abs(pf['alpha'] - a))
        dcl = pf['CL'][iff] - px['CL'][ix]
        dcd = pf['CD'][iff] - px['CD'][ix]
        dcm = pf['CM'][iff] - px['CM'][ix]
        mdcl = max(mdcl, abs(dcl))
        mdcd = max(mdcd, abs(dcd))
        mdcm = max(mdcm, abs(dcm))
        print('%6.1f %8.4f %8.4f %+9.5f %8.5f %8.5f %+9.6f %8.4f %8.4f' %
              (a, px['CL'][ix], pf['CL'][iff], dcl,
               px['CD'][ix], pf['CD'][iff], dcd,
               px['CM'][ix], pf['CM'][iff]))
    print('Ecarts max polaire : |dCL|=%.5f |dCD|=%.6f |dCM|=%.5f'
          % (mdcl, mdcd, mdcm))

    # --- Cp visqueux a A_CP (extrados, surface a surface) ---
    cx = xf.get_cp(RE, A_CP)
    cf = ff.get_cp(RE, A_CP)
    print('\n--- Cp visqueux extrados (alpha=%.0f, 0.05<x<0.95) ---' % A_CP)
    if cx is not None and cf is not None:
        print('  points xf=%d ff=%d' % (len(cx), len(cf)))
        compare_surface('Cp visqueux', cx[:, 0], cx[:, -1],
                        cf[:, 0], cf[:, -1])

    # --- Cp non visqueux (extrados) ---
    ix = xf.get_cpi(A_CP)
    iff = ff.get_cpi(A_CP)
    print('\n--- Cp non visqueux extrados (alpha=%.0f) ---' % A_CP)
    if ix is not None and iff is not None:
        print('  points xf=%d ff=%d' % (len(ix), len(iff)))
        compare_surface('Cp non visqueux', ix[:, 0], ix[:, -1],
                        iff[:, 0], iff[:, -1])

    # --- Couche limite (extrados, surface a surface, hors sillage) ---
    bx = xf.get_bl(RE, A_CP)
    bf = ff.get_bl(RE, A_CP)
    print('\n--- Couche limite extrados (alpha=%.0f) ---' % A_CP)
    if bx is not None and bf is not None:
        print('  points xf=%d ff=%d (xf inclut le sillage)'
              % (len(bx['x']), len(bf['x'])))
        compare_surface('Dstar', bx['x'], bx['Dstar'],
                        bf['x'], bf['Dstar'])
        compare_surface('Theta', bx['x'], bx['Theta'],
                        bf['x'], bf['Theta'])
        compare_surface('Ue/Vinf', bx['x'], bx['Ue_Vinf'],
                        bf['x'], bf['Ue_Vinf'])
        compare_surface('Cf', bx['x'], bx['Cf'], bf['x'], bf['Cf'])

    print('\n' + '=' * 70)
    print('Parite : polaire dCL=%.5f dCD=%.6f dCM=%.5f' % (mdcl, mdcd, mdcm))
    print('=' * 70)


if __name__ == '__main__':
    main()
