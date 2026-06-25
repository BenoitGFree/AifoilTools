#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanceur de l'interface graphique AirfoilTools."""

import sys
import os

if getattr(sys, 'frozen', False):
    # Mode PyInstaller : les modules sont dans le bundle
    _base = sys._MEIPASS
    if _base not in sys.path:
        sys.path.insert(0, _base)
else:
    # Mode developpement
    _root = os.path.dirname(os.path.abspath(__file__))
    _src = os.path.join(_root, 'sources')
    if _src not in sys.path:
        sys.path.insert(0, _src)

def _selftest():
    u"""Auto-test rapide du bundle (solveurs disponibles + calcul FlexFoil).

    Declenche par la variable d'environnement AIRFOILTOOLS_SELFTEST=1 ou
    l'argument --selftest. Permet de valider un build gele (PyInstaller)
    sans interface : verifie que les backends se chargent et qu'un calcul
    FlexFoil natif s'execute. Sort avec le code 0 si OK, 1 sinon.
    """
    try:
        from model.pipeline import available_solvers
        solvers = available_solvers()
        print('Solveurs disponibles : %s' % ', '.join(solvers))
        if 'flexfoil' in solvers:
            from model.profil import Profil
            from model.simulation import Simulation
            res = Simulation(
                Profil.from_naca('2412'),
                params={'RE': 1e6, 'ALPHA_MIN': 0.0, 'ALPHA_MAX': 4.0,
                        'ALPHA_STEP': 2.0},
                solver='flexfoil').run()
            cl = res.get_polar(1e6)['CL']
            print('FlexFoil OK : %d points, CL(0..4)=%s'
                  % (len(cl), [round(float(v), 4) for v in cl]))
            print('  cp=%s bl=%s cpi=%s'
                  % (res.has_cp, res.has_bl, res.has_cpi))
        else:
            print('FlexFoil indisponible dans ce bundle.')
        return 0
    except Exception as exc:  # pragma: no cover - diagnostic de build
        import traceback
        traceback.print_exc()
        print('SELFTEST ECHEC : %s' % exc)
        return 1


if __name__ == '__main__':
    if os.environ.get('AIRFOILTOOLS_SELFTEST') == '1' or \
            '--selftest' in sys.argv:
        sys.exit(_selftest())
    from gui.main_window import main
    main()
