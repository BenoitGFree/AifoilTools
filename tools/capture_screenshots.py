#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture des screenshots pour le manuel utilisateur AirfoilTools.

Lance la GUI et capture les ecrans cles via QWidget.grab() (capture
native Qt, ne necessite pas que la fenetre soit visible a l'ecran).

Usage :
    env_py3\\Scripts\\python.exe tools\\capture_screenshots.py
"""

import os
import sys
import time
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'sources'))

# Sur un poste sans police systeme (CI / headless Linux), fournir a Qt
# les polices TTF embarquees avec matplotlib evite le rendu en "tofu".
_FONTDIR = os.path.join(
    _ROOT, 'env_py3', 'Lib', 'site-packages',
    'matplotlib', 'mpl-data', 'fonts', 'ttf')
if os.path.isdir(_FONTDIR) and not os.environ.get('QT_QPA_FONTDIR'):
    os.environ['QT_QPA_FONTDIR'] = _FONTDIR

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


def _make_calque_asset(path):
    """Genere une image de calque synthetique (silhouette de profil).

    Simule un plan/scan de profil a decalquer : un profil NACA cambre
    (different du profil courant 2412) dessine en silhouette grise sur
    un fond couleur papier. L'image est rognee a la corde, de sorte
    qu'au chargement elle se cale sur la corde (1000 mm).
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from model.profil_spline import ProfilSpline

    prof = ProfilSpline.from_naca('4412', n_points=240)
    prof.normalize()
    pts = prof.points

    fig = Figure(figsize=(10, 4), dpi=120)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor('#efe7d3')   # teinte papier
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.fill(pts[:, 0], pts[:, 1], color='#6f6f6f', alpha=1.0, zorder=1)
    ax.plot(pts[:, 0], pts[:, 1], color='#1c1c1c', lw=1.4, zorder=2)
    ax.set_aspect('equal')
    ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
    ax.set_ylim(pts[:, 1].min() - 5, pts[:, 1].max() + 5)
    fig.savefig(path, dpi=120, facecolor='#efe7d3')


def capture_calque(app, win):
    """Capture l'onglet Profils avec une image de calque en arriere-plan."""
    print("[12] Onglet Profils - image de calque")
    tab = win._tab_profils

    # Profil courant en mode spline (points de controle visibles)
    p = tab.profil_current
    if p is not None and not p.has_splines:
        p.approximate_spline(degree=11, max_dev=0.001,
                             smoothing=0.1, max_segments=4)
        tab._canvas.set_current_profil(p)

    # Masquer la reference pour ne pas surcharger l'illustration
    win._tab_profils._chk_reference.setChecked(False)
    win._tab_profils._chk_deviation.setChecked(False)

    # Generer et charger l'image de calque
    calque = os.path.join(tempfile.gettempdir(), '_calque_demo.png')
    _make_calque_asset(calque)
    tab.load_background_image(calque)
    tab._chk_image.setChecked(True)
    tab.zoom_fit()
    pump(app, 12)
    save(win, '12_tab_profils_calque.png')

    # Nettoyage de l'etat
    tab.clear_background_image()
    win._tab_profils._chk_reference.setChecked(True)
    try:
        os.remove(calque)
    except OSError:
        pass


def generate_ba_tangent_figure():
    """Genere la figure illustrant la contrainte de tangente au BA.

    Deux panneaux : a gauche la contrainte active (tangente verticale,
    P1 aligne verticalement avec le BA) ; a droite la contrainte liberee
    (P1 deplace librement, tangente inclinee).
    """
    print("[13] Figure - contrainte tangente verticale au BA")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle
    from model.bezier import Bezier

    blue = '#1f77b4'
    grey = '#888888'

    def draw_panel(ax, cp, title):
        b = Bezier(cp)
        pts = b.points
        # Courbe
        ax.plot(pts[:, 0], pts[:, 1], '-', color=blue, lw=2.0)
        # Polygone de controle
        ax.plot(cp[:, 0], cp[:, 1], '--', color=grey, lw=0.9)
        # Points de controle interieurs (carres)
        ax.plot(cp[1:-1, 0], cp[1:-1, 1], 's', color=blue, ms=7)
        # BA (P0)
        ax.plot(cp[0, 0], cp[0, 1], 'o', color=blue, ms=6)
        ax.annotate('BA (P0)', xy=cp[0], xytext=(20, -28),
                    textcoords='offset points', fontsize=10)
        ax.annotate('P1', xy=cp[1], xytext=(14, 6),
                    textcoords='offset points', fontsize=10)
        # Cercle rouge sur P1 (rappel de la demande utilisateur)
        ax.add_patch(Circle(cp[1], radius=22, fill=False,
                            edgecolor='#d62728', lw=2.0))
        # Tangente au BA : direction P1 - P0, prolongee
        d = cp[1] - cp[0]
        d = d / (np.hypot(d[0], d[1]) + 1e-12)
        t = np.array([-40.0, 130.0])
        line = cp[0][None, :] + t[:, None] * d[None, :]
        ax.plot(line[:, 0], line[:, 1], ':', color='#d62728', lw=1.6)
        ax.set_title(title, fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlim(-60, 360)
        ax.set_ylim(-45, 150)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (mm)')

    fig = Figure(figsize=(10, 4.2), dpi=110)
    FigureCanvasAgg(fig)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    cp_active = np.array([
        [0.0, 0.0], [0.0, 75.0], [140.0, 95.0],
        [330.0, 78.0]])
    cp_free = cp_active.copy()
    cp_free[1] = [80.0, 60.0]

    draw_panel(ax1, cp_active,
               u"Contrainte active : tangente verticale")
    draw_panel(ax2, cp_free,
               u"Contrainte libérée : tangente inclinée")
    ax1.set_ylabel('y (mm)')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, '13_ba_tangente.png')
    fig.savefig(path, dpi=110)
    print(u"  -> 13_ba_tangente.png")


def main():
    print("=" * 60)
    print("Capture screenshots manuel utilisateur AirfoilTools")
    print("Sortie : %s" % OUT_DIR)
    print("=" * 60)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    win = MainWindow()
    win.resize(1280, 760)
    win.show()
    pump(app, 20)

    capture_main_window(app, win)
    capture_tab_profils(app, win)
    capture_calque(app, win)
    capture_convert_dialog(app, win)
    capture_courbure_warning(app, win)
    capture_tab_xfoil(app, win)
    capture_tab_results(app, win)

    generate_ba_tangent_figure()

    print("=" * 60)
    print("Termine.")
    print("=" * 60)

    win.close()


if __name__ == '__main__':
    main()
