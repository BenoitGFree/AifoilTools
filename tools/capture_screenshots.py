#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture des screenshots pour le manuel utilisateur AifoilTools.

Lance la GUI et capture les ecrans cles via QWidget.grab() (capture
native Qt, ne necessite pas que la fenetre soit visible a l'ecran).

Usage :
    env_py3\\Scripts\\python.exe tools\\capture_screenshots.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'sources'))

from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QMessageBox
)
from PySide6.QtCore import Qt, QTimer

from gui.main_window import MainWindow
from model.profil_spline import ProfilSpline


OUT_DIR = os.path.join(_ROOT, 'docs', 'manuel', 'images')
os.makedirs(OUT_DIR, exist_ok=True)


def save(widget, name):
    """Sauve un screenshot du widget."""
    path = os.path.join(OUT_DIR, name)
    pix = widget.grab()
    pix.save(path)
    print(u"  -> %s (%dx%d)" % (
        name, pix.width(), pix.height()))


def pump(app, n=5):
    """Force Qt a traiter les evenements en attente."""
    for _ in range(n):
        app.processEvents()
        time.sleep(0.05)


def capture_main_window(app, win):
    """Capture la fenetre principale au demarrage."""
    print("[1] Fenetre principale au demarrage")
    pump(app, 10)
    save(win, '01_main_window.png')


def capture_tab_profils(app, win):
    """Capture l'onglet Profils en differents etats."""
    print("[2] Onglet Profils - etat par defaut")
    win._tabs.setCurrentWidget(win._tab_profils)
    pump(app, 10)
    save(win, '02_tab_profils_default.png')

    print("[3] Onglet Profils - profils en mode Spline")
    p = win._tab_profils.profil_current
    if p is not None and not p.has_splines:
        p.approximate_spline(degree=11, max_dev=0.001,
                             smoothing=0.1, max_segments=4)
        win._tab_profils._canvas.set_current_profil(p)
    pump(app, 10)
    save(win, '03_tab_profils_spline.png')

    print("[4] Onglet Profils - avec porcupines de courbure")
    win._tab_profils._chk_porc_current.setChecked(True)
    pump(app, 10)
    save(win, '04_tab_profils_courbure.png')

    print("[5] Onglet Profils - avec points echantillonnes")
    win._tab_profils._chk_porc_current.setChecked(False)
    win._tab_profils._chk_sample_pts.setChecked(True)
    pump(app, 10)
    save(win, '05_tab_profils_pts.png')

    print("[6] Onglet Profils - deviation")
    # Convertir aussi la reference
    p_ref = win._tab_profils.profil_reference
    if p_ref is not None and not p_ref.has_splines:
        p_ref.approximate_spline(degree=11, max_dev=0.001,
                                 smoothing=0.1, max_segments=4)
        win._tab_profils._canvas.set_reference_profil(p_ref)
    win._tab_profils._chk_sample_pts.setChecked(False)
    win._tab_profils._chk_deviation.setChecked(True)
    pump(app, 10)
    save(win, '06_tab_profils_deviation.png')

    # Reset
    win._tab_profils._chk_deviation.setChecked(False)


def capture_convert_dialog(app, win):
    """Capture le dialogue 'Convertir en Spline'."""
    print("[7] Dialogue Convertir en Spline")
    dlg = QDialog(win)
    dlg.setWindowTitle("Convertir en Spline")
    form = QFormLayout(dlg)

    spn_ext = QSpinBox()
    spn_ext.setRange(2, 30)
    spn_ext.setValue(11)
    form.addRow(u"Degré extrados :", spn_ext)

    spn_int = QSpinBox()
    spn_int.setRange(2, 30)
    spn_int.setValue(11)
    form.addRow(u"Degré intrados :", spn_int)

    spn_tol = QDoubleSpinBox()
    spn_tol.setRange(0.0001, 0.1)
    spn_tol.setDecimals(4)
    spn_tol.setSingleStep(0.0005)
    spn_tol.setValue(0.001)
    form.addRow(u"Tolérance :", spn_tol)

    spn_max_seg = QSpinBox()
    spn_max_seg.setRange(1, 20)
    spn_max_seg.setValue(4)
    form.addRow("Max segments :", spn_max_seg)

    spn_smooth = QDoubleSpinBox()
    spn_smooth.setRange(0.0, 1.0)
    spn_smooth.setDecimals(2)
    spn_smooth.setSingleStep(0.01)
    spn_smooth.setValue(0.1)
    form.addRow("Lissage :", spn_smooth)

    buttons = QDialogButtonBox(
        QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    form.addRow(buttons)

    dlg.adjustSize()
    dlg.show()
    pump(app, 10)
    save(dlg, '07_dialog_convert.png')
    dlg.close()


def capture_courbure_warning(app, win):
    """Capture le QMessageBox courbure indisponible."""
    print("[8] Message d'avertissement courbure")
    box = QMessageBox(win)
    box.setIcon(QMessageBox.Information)
    box.setWindowTitle(u"Courbure indisponible")
    box.setText(
        u"La courbure ne peut etre tracee que sur des profils "
        u"definis par des Beziers (splines).\n\n"
        u"Solutions :\n"
        u"  - Convertir le profil courant via le menu "
        u"« Edition › Convertir en Spline »\n"
        u"  - Charger un profil au format .bspl ou .bez")
    box.show()
    pump(app, 10)
    save(box, '08_msg_courbure.png')
    box.close()


def capture_tab_xfoil(app, win):
    """Capture l'onglet Parametrage XFoil."""
    print("[9] Onglet Parametrage XFoil")
    win._tabs.setCurrentWidget(win._tab_xfoil)
    pump(app, 10)
    save(win, '09_tab_xfoil.png')


def capture_tab_results(app, win):
    """Capture l'onglet Resultats (vide)."""
    print("[10] Onglet Resultats - vide")
    win._tabs.setCurrentWidget(win._tab_results)
    pump(app, 10)
    save(win, '10_tab_results_empty.png')


def main():
    print("=" * 60)
    print("Capture screenshots manuel utilisateur AifoilTools")
    print("Sortie : %s" % OUT_DIR)
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.resize(1280, 760)
    win.show()
    pump(app, 20)

    capture_main_window(app, win)
    capture_tab_profils(app, win)
    capture_convert_dialog(app, win)
    capture_courbure_warning(app, win)
    capture_tab_xfoil(app, win)
    capture_tab_results(app, win)

    print("=" * 60)
    print("Termine.")
    print("=" * 60)

    win.close()


if __name__ == '__main__':
    main()
