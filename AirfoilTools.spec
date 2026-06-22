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
    """Recompile le manuel LaTeX avant le bundling.

    Garantit que le PDF embarque (docs/manuel/manuel.pdf) reflete la
    derniere source manuel.tex (notamment la version \\manuelversion).
    Si latexmk est absent ou echoue, on conserve le PDF existant en
    affichant un avertissement (le build n'est pas interrompu).
    """
    manual_dir = os.path.join(SPECPATH, 'docs', 'manuel')
    latexmk = shutil.which('latexmk')
    if latexmk is None:
        print("[spec] AVERTISSEMENT : latexmk introuvable, "
              "le manuel ne sera PAS recompile (PDF existant conserve).")
        return
    print("[spec] Recompilation du manuel LaTeX (latexmk)...")
    try:
        subprocess.run(
            [latexmk, '-pdf', '-interaction=nonstopmode', 'manuel.tex'],
            cwd=manual_dir, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[spec] Manuel recompile : docs/manuel/manuel.pdf")
    except subprocess.CalledProcessError as exc:
        print("[spec] AVERTISSEMENT : echec de la compilation du manuel "
              "(code %s), PDF existant conserve." % exc.returncode)


_inject_version()
_rebuild_manual()

a = Analysis(
    ['run_gui.py'],
    pathex=['sources'],
    binaries=[],
    datas=[
        ('sources/model/defaults_xfoil.cfg', 'model'),
        # Manuel utilisateur (accessible via menu Aide)
        ('docs/manuel/manuel.pdf', 'docs'),
        # Executable XFoil (necessaire aux simulations)
        ('externaltools/xfoil/xfoil.exe', 'externaltools/xfoil'),
    ],
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
        'gui.simulation_worker',
        'gui.dialog_uiuc',
        # matplotlib backend pour PySide6
        'matplotlib.backends.backend_qtagg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
