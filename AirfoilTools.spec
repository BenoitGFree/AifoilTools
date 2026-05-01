# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for AirfoilTools."""

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
