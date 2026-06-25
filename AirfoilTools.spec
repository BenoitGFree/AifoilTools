# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for AirfoilTools."""

import os
import shutil
import subprocess
import sys


def _inject_version():
    """Genere docs/manuel/version.tex depuis __version__ (source unique).

    Delegue a docs/manuel/gen_version.py pour ne pas dupliquer la
    logique : le meme script est lancable a la main avant une
    compilation LaTeX isolee. version.tex est un artefact (gitignore),
    regenere ici a chaque build.
    """
    gen = os.path.join(SPECPATH, 'docs', 'manuel', 'gen_version.py')
    try:
        subprocess.run([sys.executable, gen], check=True)
        print("[spec] version.tex genere depuis __version__.")
    except subprocess.CalledProcessError as exc:
        print("[spec] AVERTISSEMENT : generation de version.tex echouee "
              "(code %s)." % exc.returncode)


def _rebuild_manual():
    """Recompile les manuels LaTeX (FR + EN) avant le bundling.

    Garantit que les PDF embarques (docs/manuel/manuel.pdf et
    manuel_en.pdf) refletent les dernieres sources (notamment la
    version \\manuelversion injectee). Si latexmk est absent ou echoue,
    on conserve le PDF existant en affichant un avertissement (le build
    n'est pas interrompu).
    """
    manual_dir = os.path.join(SPECPATH, 'docs', 'manuel')
    latexmk = shutil.which('latexmk')
    if latexmk is None:
        print("[spec] AVERTISSEMENT : latexmk introuvable, "
              "les manuels ne seront PAS recompiles (PDF existants conserves).")
        return
    for tex in ('manuel.tex', 'manuel_en.tex'):
        if not os.path.isfile(os.path.join(manual_dir, tex)):
            continue
        print("[spec] Recompilation du manuel LaTeX (%s)..." % tex)
        try:
            subprocess.run(
                [latexmk, '-pdf', '-interaction=nonstopmode', tex],
                cwd=manual_dir, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[spec] Manuel recompile : %s" % tex.replace('.tex', '.pdf'))
        except subprocess.CalledProcessError as exc:
            print("[spec] AVERTISSEMENT : echec de la compilation de %s "
                  "(code %s), PDF existant conserve." % (tex, exc.returncode))


_inject_version()
_rebuild_manual()


def _collect_flexfoil():
    """Collecte le backend FlexFoil s'il est installe (dependance optionnelle).

    FlexFoil (https://foil.flexcompute.com, MIT) est une bibliotheque
    native : il faut embarquer le module Rust ``_rustfoil.*.pyd`` et les
    wrappers Python. Retourne (binaries, hiddenimports). Si flexfoil
    n'est pas installe dans l'environnement de build, retourne ([], []) :
    le bundle se construit alors sans ce backend (XFoil reste disponible).
    """
    try:
        import flexfoil  # noqa: F401
    except Exception:
        print("[spec] FlexFoil absent de l'environnement : backend non "
              "embarque (XFoil seul).")
        return [], []
    import glob as _glob
    ff_dir = os.path.dirname(flexfoil.__file__)
    # Module natif Rust : collecte explicite (collect_dynamic_libs /
    # collect_all ne reperent pas les .pyd d'extension). Place sous
    # flexfoil/ pour rester importable comme flexfoil._rustfoil.
    pyds = _glob.glob(os.path.join(ff_dir, '_rustfoil*.pyd'))
    binaries = [(p, 'flexfoil') for p in pyds]
    # On liste seulement les wrappers reellement utilises (pas
    # flexfoil.server, qui depend de starlette/uvicorn non installes).
    hidden = [
        'model.flexfoil_backend',
        'flexfoil', 'flexfoil.airfoil', 'flexfoil.database',
        'flexfoil.polar', 'flexfoil._rustfoil',
    ]
    print("[spec] FlexFoil embarque (%d module(s) natif(s) : %s)."
          % (len(binaries), ', '.join(os.path.basename(p) for p in pyds)))
    return binaries, hidden


_ff_binaries, _ff_hidden = _collect_flexfoil()

# Manuels a embarquer : FR toujours, EN seulement s'il a ete compile
# (manuel_en.pdf). Le menu Aide ouvre la version correspondant a la
# langue de l'interface, avec fallback FR.
_datas = [
    ('sources/model/defaults_xfoil.cfg', 'model'),
    ('sources/model/defaults_flexfoil.cfg', 'model'),
    # Manuel utilisateur (accessible via menu Aide)
    ('docs/manuel/manuel.pdf', 'docs'),
    # Executable XFoil (necessaire aux simulations)
    ('externaltools/xfoil/xfoil.exe', 'externaltools/xfoil'),
]
if os.path.isfile(os.path.join(SPECPATH, 'docs', 'manuel', 'manuel_en.pdf')):
    _datas.append(('docs/manuel/manuel_en.pdf', 'docs'))

a = Analysis(
    ['run_gui.py'],
    pathex=['sources'],
    binaries=_ff_binaries,
    datas=_datas,
    hiddenimports=[
        # model package
        'model',
        'model.base',
        'model.bezier',
        'model.bezier_spline',
        'model.profil',
        'model.profil_spline',
        'model.analyse',
        'model.simulation',
        'model.pipeline',
        'model.foilconfig',
        'model.xfoil_preprocessor',
        'model.xfoil_simulator',
        'model.xfoil_postprocessor',
        'model.uiuc_loader',
        # gui package
        'gui',
        'gui.main_window',
        'gui.tab_profils',
        'gui.profil_canvas',
        'gui.tab_xfoil',
        'gui.tab_results',
        'gui.result_cell',
        'gui.tab_cp',
        'gui.simulation_worker',
        'gui.dialog_uiuc',
        # matplotlib backend pour PySide6
        'matplotlib.backends.backend_qtagg',
    ] + _ff_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # plotly / pandas sont des dependances optionnelles de flexfoil,
    # importees paresseusement (graphiques .plot() / export DataFrame)
    # jamais utilisees par AirfoilTools : on les exclut du bundle.
    excludes=['plotly', 'pandas'],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AirfoilTools',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # pas de console noire
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AirfoilTools',
)
