# Spike — FlexFoil en remplacement de XFoil.exe

Évaluation de [FlexFoil](https://foil.flexcompute.com/) (réimplémentation Rust
du solveur XFoil de Mark Drela, par Flexcompute) comme moteur de calcul à la
place de l'exécutable `xfoil.exe` actuellement embarqué dans
`externaltools/xfoil/`.

Branche : `spike/flexfoil-evaluation`. **Ce dossier est un spike jetable** : il
ne modifie pas le code produit, il documente une décision.

## TL;DR

| Verrou | Statut | Détail |
|--------|--------|--------|
| **Licence** | ✅ **MIT** | `flexcompute/flexfoil`, compatible LGPL-3.0, redistribution dans l'installateur autorisée. |
| **Validation numérique** | ✅ **Identique à XFoil** | Écarts max sur NACA 2412, Re=1e6, α∈[−4°,12°] : ΔCL=5e−4, ΔCD=1e−5, ΔCM=1e−4 (bruit numérique). Transition extrados/intrados identique à 1e−3 près. |
| **Parité fonctionnelle** | ✅ **Complète** | Cp, Cp non visqueux (`viscous=False`), couche limite **déjà séparée par face**, transition, volet (`with_flap`), chargement coords/.dat, `ncrit`/`mach`/`xstrip` configurables. |
| **Packaging** | ⚠️ **Contrainte Python 3.11** | Wheels binaires **cp311 uniquement** (pas de cp312/cp313). Le projet tourne en **3.13**. |

**Conclusion** : techniquement FlexFoil est un remplaçant supérieur (API
structurée, fin du parsing de fichiers texte, fin de la reconstruction
topologique extrados/intrados, Cp non visqueux trivial). **Le seul vrai
obstacle est la version de Python.**

## Détail des résultats

Voir [`resultats_comparaison.txt`](resultats_comparaison.txt) pour la sortie
brute. Reproductible via [`compare_xfoil_flexfoil.py`](compare_xfoil_flexfoil.py)
(doit tourner sous un interpréteur **Python 3.11** où `flexfoil` est installé).

### Coefficients (extrait)

```
 alpha    CL_xf    CL_ff       dCL    CD_xf    CD_ff       dCD    CM_xf    CM_ff
   0.0   0.2372   0.2370  -0.00017  0.00565  0.00564 -6e-06   -0.0520  -0.0520
   6.0   0.9020   0.9019  -0.00012  0.00905  0.00905 +3e-06   -0.0505  -0.0505
  12.0   1.4085   1.4090  +0.00046  0.02001  0.02001 +1e-06   -0.0264  -0.0265
```

C'est attendu : FlexFoil est un portage fidèle du même algorithme, pas un
solveur différent.

## API FlexFoil (vérifiée, v1.1.6)

```python
import flexfoil
foil = flexfoil.naca("2412")                       # ou .load("x.dat") / .from_coordinates(x,y,name,n_panels)
r = foil.solve(alpha=6.0, Re=1e6, ncrit=9.0,       # mach, viscous, max_iter, xstrip_upper/lower...
               max_iter=200)
r.cl, r.cd, r.cm, r.ld, r.converged
r.cp, r.cp_x, r.x_tr_upper, r.x_tr_lower

inv = foil.solve(alpha=6.0, Re=1e6, viscous=False) # Cp non visqueux -> remplace le hack "session XFoil séparée"

bl = foil.bl_distribution(alpha=6.0, Re=1e6)       # BLResult déjà séparé par face :
bl.x_upper, bl.delta_star_upper, bl.theta_upper, bl.cf_upper, bl.h_upper, bl.ue_upper
bl.x_lower, ...                                     # idem intrados

flap = foil.with_flap(hinge_x=0.75, deflection=10.0)
pol  = foil.polar(alpha=(-5, 15, 0.5), Re=1e6)     # balayage parallélisé
```

**Gain d'architecture** : `bl.*_upper`/`*_lower` sont déjà découpés. Tout le
travail de reconstruction extrados/intrados (ordre DUMP, ordre Selig, tri par
abscisse curviligne, retrait du sillage, méthode topologique) — la source des
bugs récents — **disparaît**.

## Packaging — le point dur

- Le solveur natif est un **unique fichier `_rustfoil.cp311-win_amd64.pyd`
  (~828 Ko)**, Rust autonome, sans DLL externe. Wrappers Python purs à côté.
- `solve()` n'importe **ni numpy ni plotly** (chargés paresseusement, seulement
  pour `.plot()`/export pandas) → bundle PyInstaller minimal possible
  (le `.pyd` + 4 `.py`, exclure plotly).
- **MAIS** : PyPI ne publie des wheels que pour **cp311** (+ macOS x86_64/arm64,
  Linux cp38). **Aucun wheel cp312/cp313.** En Python 3.13, `pip install
  flexfoil` retombe sur le sdist et tente de **compiler du Rust** (télécharge
  `rustup`) — inacceptable pour un produit livré en installateur.

### Options pour lever le verrou Python

1. **Rétrograder le projet en Python 3.11.** Simple, wheel officiel, mais touche
   tout l'environnement (`env_py3`, `.spec`, CI). À évaluer vis-à-vis de PySide6
   / autres deps.
2. **Compiler nous-mêmes un wheel cp313** depuis le sdist MIT (maturin + Rust),
   une fois, et l'héberger (vendored). Maîtrise totale, mais ajoute une chaîne
   de build Rust à maintenir.
3. **Attendre** des wheels cp312/cp313 amont (projet actif, push récent) avant
   d'adopter.

## Recommandation

Adopter FlexFoil **comme second backend optionnel** derrière l'abstraction
`AbstractSimulator` / pipeline déjà en place, **conditionné à la résolution du
verrou Python 3.11** (option 1 ou 2 ci-dessus). Garder XFoil.exe en repli tant
que le packaging 3.13 n'est pas tranché.

Le code d'adaptation serait modeste : un moteur qui appelle `solve()` /
`bl_distribution()` et remplit la structure `SimulationResults` existante
(`cp=[x,y,Cp]`, `bl[re][alpha]`, `cpi[alpha]`). Il **supprime** l'essentiel de
`xfoil_preprocessor.py`, `xfoil_simulator.py` et `xfoil_postprocessor.py`.

## Comment rejouer le spike

```bash
# Obtenir un Python 3.11 jetable (proxy d'entreprise -> --native-tls obligatoire)
env_py3/Scripts/python.exe -m pip install uv
export UV_NATIVE_TLS=1
env_py3/Scripts/python.exe -m uv venv tmp_ff311 --python 3.11 --native-tls
env_py3/Scripts/python.exe -m uv pip install --python tmp_ff311/Scripts/python.exe --native-tls flexfoil numpy
tmp_ff311/Scripts/python.exe spikes/flexfoil/compare_xfoil_flexfoil.py
```
