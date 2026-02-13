#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Demonstration : comparaison de deux simulations XFoil reelles.

Profil A (reference) : NACA 0012  (rouge)
Profil B (a comparer) : NACA 2412  (bleu, couleur par defaut)

Deroule le pipeline complet :
  1. Creation des profils via Profil.from_naca()
  2. Simulation XFoil reelle (preprocessing + execution + parsing)
  3. Affichage des resultats numeriques
  4. Comparaison graphique via Analyse (polaires + Cp)

Usage::

    python sources/model/aerodynamique/foil2d/test.py

@author: Nervures
@date: 2026-02
"""

import os
import sys
import logging

_here = os.path.dirname(os.path.abspath(__file__))
_pkg = os.path.normpath(os.path.join(_here, '..', 'airfoiltools'))
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)

# Logging visible en console
logging.basicConfig(
    level=logging.INFO,
    format='%(name)-20s %(levelname)-8s %(message)s'
)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from profil import Profil
from simulation import Simulation
from analyse import Analyse


# =====================================================================
#  Parametres de simulation
# =====================================================================

PARAMS = {
    'RE_LIST': [300000, 500000, 1000000],
    'MACH': 0.0,
    'ALPHA_MIN': 0.0,
    'ALPHA_MAX': 5.0,
    'ALPHA_STEP': 0.5,
    'VISCOUS': True,
    'NCRIT': 9,
    'ITER': 200,
    'REPANEL': True,
    'NPANEL': 200,
    'TIMEOUT': 300,
}


# =====================================================================
#  Creation des profils et simulations
# =====================================================================

print("=" * 60)
print("Test foil2d - Process complet avec XFoil")
print("=" * 60)

print("\n--- 1. Creation des profils ---")
profil_a = Profil.from_naca('0012')
profil_b = Profil.from_naca('2412')

print("Profil A (reference) : %s" % profil_a)
print("Profil B (compare)   : %s" % profil_b)


# =====================================================================
#  Execution des simulations
# =====================================================================

print("\n--- 2. Simulation XFoil du profil A (%s) ---" % profil_a.name)
sim_a = Simulation(profil_a, params=PARAMS)
print("  Repertoire de travail : %s" % sim_a.work_dir)
sim_a.run()

print("\n--- 3. Simulation XFoil du profil B (%s) ---" % profil_b.name)
sim_b = Simulation(profil_b, params=PARAMS)
print("  Repertoire de travail : %s" % sim_b.work_dir)
sim_b.run()


# =====================================================================
#  Post-traitement via Analyse
# =====================================================================

print("\n--- 4. Analyse comparative ---")
analyse = Analyse()
analyse.add(sim_a, reference=True)
analyse.add(sim_b)

print(repr(analyse))
analyse.summary()


# =====================================================================
#  Traces comparatifs
# =====================================================================

print("\n--- 5. Traces ---")
analyse.plot_polars(show=False)
analyse.plot_cp(5.0, show=False)


# =====================================================================
#  Fichiers generes
# =====================================================================

print("\n--- 6. Fichiers generes ---")
for sim in [sim_a, sim_b]:
    print("\n  %s : %s" % (sim.profil.name, sim.work_dir))
    for f in sorted(os.listdir(sim.work_dir)):
        size = os.path.getsize(os.path.join(sim.work_dir, f))
        print("    %-35s %8d octets" % (f, size))

print("\n  (repertoires conserves pour inspection)")

print("\n" + "=" * 60)
print("TEST TERMINE AVEC SUCCES")
print("=" * 60)

plt.show()
