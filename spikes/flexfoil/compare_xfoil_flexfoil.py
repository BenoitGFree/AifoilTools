#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""Spike de validation : FlexFoil vs XFoil.exe.

Compare les coefficients (CL, CD, CM, transition) produits par FlexFoil
(reimplementation Rust de XFoil) et par l'executable XFoil historique
embarque dans externaltools/xfoil/, sur un balayage d'incidences pour
le profil NACA 2412 a Re = 1e6.

Verifie aussi :
  - le mode non-visqueux (Cp non visqueux),
  - le chargement depuis coordonnees / fichier .dat,
  - la geometrie avec volet (with_flap),
  - la disponibilite des grandeurs de couche limite par face.

IMPORTANT : FlexFoil ne fournit (v1.1.6) que des wheels cp311. Ce script
doit donc tourner sous un interpreteur Python 3.11 ou flexfoil est
installe. XFoil.exe est appele en sous-processus comme dans le produit.

Usage :
    python compare_xfoil_flexfoil.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..', '..'))
XFOIL = os.path.join(ROOT, 'externaltools', 'xfoil', 'xfoil.exe')

NACA = '2412'
RE = 1.0e6
ALPHAS = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def run_xfoil_polar(naca, re, alphas, workdir):
    u"""Lance XFoil.exe en mode accumulation de polaire.

    :returns: dict alpha -> (cl, cd, cm, xtr_top, xtr_bot)
    """
    polar = os.path.join(workdir, 'polar.txt')
    if os.path.exists(polar):
        os.remove(polar)
    lines = ['NACA %s' % naca, 'OPER', 'VISC %d' % int(re), 'ITER 200',
             'PACC', 'polar.txt', '']
    for a in alphas:
        lines.append('ALFA %g' % a)
    lines += ['PACC', '', 'QUIT', '']
    cmd = '\n'.join(lines)
    with open(os.path.join(workdir, 'cmd.txt'), 'w') as f:
        f.write(cmd)
    proc = subprocess.Popen([XFOIL], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=workdir)
    proc.communicate(input=cmd.encode('utf-8'), timeout=120)

    out = {}
    with open(polar) as f:
        started = False
        for line in f:
            if line.lstrip().startswith('------'):
                started = True
                continue
            if not started:
                continue
            p = line.split()
            if len(p) >= 7:
                a = float(p[0])
                out[round(a, 3)] = (float(p[1]), float(p[2]),
                                    float(p[4]), float(p[5]), float(p[6]))
    return out


def run_flexfoil_polar(naca, re, alphas):
    u"""Lance FlexFoil sur les memes incidences.

    :returns: dict alpha -> (cl, cd, cm, xtr_top, xtr_bot, converged)
    """
    import flexfoil
    foil = flexfoil.naca(naca)
    out = {}
    for a in alphas:
        r = foil.solve(alpha=a, Re=re, ncrit=9.0, max_iter=200)
        out[round(a, 3)] = (r.cl, r.cd, r.cm, r.x_tr_upper, r.x_tr_lower,
                            r.converged)
    return out


def main():
    import flexfoil

    print('=' * 72)
    print('SPIKE FlexFoil vs XFoil  -  NACA %s, Re = %.0e' % (NACA, RE))
    print('flexfoil', getattr(flexfoil, '__version__', '?'),
          '| python', sys.version.split()[0])
    print('xfoil.exe :', XFOIL, '(present:', os.path.isfile(XFOIL), ')')
    print('=' * 72)

    workdir = os.path.join(HERE, '_xfoil_tmp')
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    xf = run_xfoil_polar(NACA, RE, ALPHAS, workdir)
    ff = run_flexfoil_polar(NACA, RE, ALPHAS)

    print('\n--- Coefficients : XFoil (ref) vs FlexFoil ---\n')
    hdr = ('alpha', 'CL_xf', 'CL_ff', 'dCL', 'CD_xf', 'CD_ff', 'dCD',
           'CM_xf', 'CM_ff', 'dCM')
    print('%6s %8s %8s %9s %8s %8s %9s %8s %8s %9s' % hdr)
    max_dcl = max_dcd = max_dcm = 0.0
    for a in ALPHAS:
        k = round(a, 3)
        if k not in xf or k not in ff:
            print('%6.1f  (manquant : xf=%s ff=%s)' % (a, k in xf, k in ff))
            continue
        clx, cdx, cmx = xf[k][0], xf[k][1], xf[k][2]
        clf, cdf, cmf = ff[k][0], ff[k][1], ff[k][2]
        dcl, dcd, dcm = clf - clx, cdf - cdx, cmf - cmx
        max_dcl = max(max_dcl, abs(dcl))
        max_dcd = max(max_dcd, abs(dcd))
        max_dcm = max(max_dcm, abs(dcm))
        print('%6.1f %8.4f %8.4f %+9.5f %8.5f %8.5f %+9.6f '
              '%8.4f %8.4f %+9.5f'
              % (a, clx, clf, dcl, cdx, cdf, dcd, cmx, cmf, dcm))

    print('\nEcarts max : |dCL|=%.5f  |dCD|=%.6f  |dCM|=%.5f'
          % (max_dcl, max_dcd, max_dcm))

    print('\n--- Transition (x/c) : XFoil vs FlexFoil ---\n')
    print('%6s %10s %10s %10s %10s' % ('alpha', 'top_xf', 'top_ff',
                                       'bot_xf', 'bot_ff'))
    for a in ALPHAS:
        k = round(a, 3)
        if k in xf and k in ff:
            print('%6.1f %10.4f %10.4f %10.4f %10.4f'
                  % (a, xf[k][3], ff[k][3], xf[k][4], ff[k][4]))

    # --- Mode non-visqueux (Cp non visqueux) ---
    print('\n--- Cp non visqueux (viscous=False) ---')
    foil = flexfoil.naca(NACA)
    inv = foil.solve(alpha=6.0, Re=RE, viscous=False)
    print('inviscid solve : converged=%s  n_cp=%d  cp[min]=%.3f'
          % (inv.converged, len(inv.cp), min(inv.cp)))

    # --- Couche limite par face ---
    print('\n--- Couche limite (bl_distribution) ---')
    bl = foil.bl_distribution(alpha=6.0, Re=RE)
    print('extrados : %d pts | intrados : %d pts'
          % (len(bl.x_upper), len(bl.x_lower)))
    print('grandeurs : delta*, theta, cf, h, ue  (deja separees upper/lower)')

    # --- Chargement coordonnees / .dat ---
    print('\n--- Chargement geometrie ---')
    x = [p[0] for p in foil.raw_coords]
    y = [p[1] for p in foil.raw_coords]
    foil2 = flexfoil.from_coordinates(x, y, name='roundtrip', n_panels=160)
    r2 = foil2.solve(alpha=6.0, Re=RE)
    print('from_coordinates : CL=%.4f (vs %.4f)' % (r2.cl, ff[6.0][0]))

    # --- Volet ---
    print('\n--- Geometrie avec volet (with_flap) ---')
    try:
        import inspect
        print('with_flap sig :', inspect.signature(foil.with_flap))
        flap = foil.with_flap(hinge_x=0.75, deflection=10.0)
        rf = flap.solve(alpha=6.0, Re=RE)
        print('volet 10deg : CL=%.4f (vs %.4f sans volet)'
              % (rf.cl, ff[6.0][0]))
    except Exception as e:
        print('with_flap : ECHEC ->', repr(e))

    print('\n' + '=' * 72)
    print('CONCLUSION : ecart max CL=%.5f CD=%.6f CM=%.5f' %
          (max_dcl, max_dcd, max_dcm))
    print('=' * 72)


if __name__ == '__main__':
    main()
