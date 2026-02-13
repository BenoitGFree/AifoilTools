# CLAUDE.md

## Project Overview

**AifoilTools** est une bibliothèque Python d'outils d'analyse aérodynamique 2D de profils d'aile, extraite du projet Axile (logiciel de conception de parapentes par Nervures).

- **Repository** : https://github.com/BenoitGFree/AifoilTools
- **Licence** : LGPL-3.0
- **Origine** : `sources/model/aerodynamique/foil2d/` du projet Axile

## Python Environment

- **Python** : 3.10+ (venv local `env_py3/`)
- **Environnement virtuel** : `env_py3/` (créé via `py -3 -m venv env_py3`)
- **Dépendances** : numpy, matplotlib, scipy, PySide6 (voir `requirements.txt`)
- **Activation** : `env_py3\Scripts\activate` (Windows)

## Architecture

```
AifoilTools/
├── sources/
│   ├── __init__.py
│   ├── model/                    # Package calcul (pip: airfoiltools)
│   │   ├── __init__.py           # Exports publics (imports relatifs)
│   │   ├── bezier.py             # Courbes de Bézier 2D de degré arbitraire
│   │   ├── profil.py             # Profil aérodynamique (points, NACA, Bézier)
│   │   ├── analyse.py            # Comparaison multi-profils
│   │   ├── simulation.py         # Orchestration simulations XFoil
│   │   ├── pipeline.py           # Pipeline extensible pre/sim/post
│   │   ├── base.py               # Classes abstraites (ABC)
│   │   ├── foilconfig.py         # Chargement config solveurs
│   │   ├── xfoil_preprocessor.py
│   │   ├── xfoil_simulator.py
│   │   ├── xfoil_postprocessor.py
│   │   └── defaults_xfoil.cfg    # Paramètres XFoil par défaut
│   └── gui/                      # GUI PySide6 + matplotlib
│       ├── __init__.py
│       ├── __main__.py            # python -m gui (depuis sources/)
│       ├── main_window.py         # QMainWindow + menus + onglets
│       ├── tab_profils.py         # Onglet Profils (contrôles + canvas)
│       ├── profil_canvas.py       # Canvas matplotlib interactif
│       ├── tab_xfoil.py           # Onglet Paramétrage XFoil
│       ├── tab_results.py         # Onglet Résultats (grille de ResultCell)
│       ├── result_cell.py         # Cellule d'analyse (18 types, canvas interactif)
│       └── simulation_worker.py   # QThread pour simulations non-bloquantes
├── tests/
│   ├── test_bezier.py             # 174 tests
│   ├── test_profil.py             # 37 tests
│   ├── test_simulation.py         # 33 tests
│   ├── test_foil2d.py             # Tests pipeline (sans XFoil)
│   ├── test_foil2d_xfoil.py       # Tests avec XFoil réel
│   └── test.py                    # Démo end-to-end
├── run_gui.py                     # Lanceur GUI
├── requirements.txt               # Dépendances Python 3
└── setup.py                       # pip install (package_dir mapping)
```

## Running

```bash
# Lancer la GUI
env_py3\Scripts\python.exe run_gui.py

# Tests (avec le venv Python 3)
env_py3\Scripts\python.exe -c "import unittest; import sys; sys.path.insert(0, 'tests'); from test_bezier import *; unittest.main(module='test_bezier', exit=False, verbosity=2)"
env_py3\Scripts\python.exe -c "import unittest; import sys; sys.path.insert(0, 'tests'); from test_profil import *; unittest.main(module='test_profil', exit=False, verbosity=2)"
env_py3\Scripts\python.exe -c "import unittest; import sys; sys.path.insert(0, 'tests'); from test_simulation import *; unittest.main(module='test_simulation', exit=False, verbosity=2)"
env_py3\Scripts\python.exe tests/test_foil2d.py
```

## Key Classes

### Bezier (`sources/model/bezier.py`)

Courbe de Bézier 2D de degré arbitraire.

- **Construction** : `Bezier(control_points)` ou `Bezier(target_points, degree=5)` (approximation)
- **Évaluation** : De Casteljau, `evaluate(t)`, `derivative()`, `second_derivative()`
- **Géométrie** : `tangent(t)`, `normal(t)`, `curvature(t)`
- **Points de contrôle** : `cpoint(i)`, `start_cpoint`, `end_cpoint`, `translate_cpoint(i, v)`
- **Transformations** : `translate(dx, dy)`, `rotate(angle_deg)`, `scale(factor)`, `reverse()`
- **Degré** : `elevate(times)`, `reduce(times)`
- **Approximation** : `approximate(points, degree)`, `max_deviation(points)`, `degree='find'`
- **Échantillonnage** : `sample(n)`, modes `curvilinear` et `adaptive`
- **Properties cachées** : `points`, `tangents`, `normals`, `curvatures`
- **Cache** : `_cache = {}`, invalidation à deux niveaux (géométrie / échantillonnage)

### Profil (`sources/model/profil.py`)

Profil aérodynamique 2D, stockage en millimètres, convention Selig.

- **Convention Selig** : BF → extrados (x décroissant) → BA → intrados (x croissant) → BF
- **Création** : `Profil(points)`, `Profil.from_naca('2412')`, `Profil.from_file(path)`
- **Géométrie** : `leading_edge`, `trailing_edge`, `chord`, `calage`, `relative_thickness`, `relative_camber`
- **Extrados/Intrados** : `extrados`, `intrados` (BA → BF, x croissant)
- **Transformations** : `scale()`, `rotate()`, `translate()`, `normalize()`
- **Mode Bézier** :
  - `approximate_bezier(degree, max_dev=None, n_points=None)` : active le mode
  - `has_beziers` : True si défini par Béziers
  - `bezier_extrados`, `bezier_intrados` : accès aux objets Bezier
  - `clear_beziers()` : retour au mode discret
  - `points` setter efface automatiquement les Béziers
- **I/O** : `write(path, fmt)`, formats selig/lednicer/csv

### Simulation / Analyse (`sources/model/simulation.py`, `analyse.py`)

- **Simulation** : orchestre le pipeline XFoil pour un profil + paramètres
- **SimulationResults** : résultats structurés, accès `get_polar(re)`, `cl_max(re)`, `finesse_max(re)`
- **Analyse** : conteneur multi-simulations, comparaison graphique (référence en rouge)

## Design Patterns

### Cache Bezier (à appliquer partout)

```python
_cache = {}
def _invalidate(self, geometry=True):
    if geometry:
        self._cache.clear()          # tout recalculer
    else:
        self._cache.pop('points', None)  # seul l'échantillonnage change
```

- Les setters n'invalident que si la valeur change réellement
- Les properties retournent `.copy()` des données cachées

### Conventions de nommage

- Points de contrôle : préfixe `cpoint` (`cpoint(i)`, `start_cpoint`, `end_cpoint`)
- `points` = points échantillonnés (pas les points de contrôle)
- Transformations retournent `self` (chaînage)

### Imports

- **Dans sources/model/** : imports relatifs explicites (`from .bezier import Bezier`)
- **Dans sources/gui/** : imports absolus depuis `model` (`from model.profil import Profil`)
- **Dans tests/** : `sys.path.insert(0, 'sources')` puis `from model.bezier import Bezier`
- **pip install** : `package_dir={'airfoiltools': 'sources/model'}` → `from airfoiltools import Bezier`

## Projet parent Axile

Le code d'AifoilTools est aussi présent dans Axile sous `sources/model/aerodynamique/foil2d/`. Les deux copies évoluent indépendamment pour l'instant. Le worktree Axile est dans `C:\Liclipse\workspace\Axile_worktrees\57-integration-xfoil`.
