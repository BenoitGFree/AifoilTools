#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture des screenshots pour le manuel utilisateur AirfoilTools.

Lance la GUI et capture les ecrans cles via QWidget.grab() (capture
native Qt, ne necessite pas que la fenetre soit visible a l'ecran).
Certaines figures explicatives sont generees directement avec
matplotlib (Agg).

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
    QSpinBox, QDoubleSpinBox, QMessageBox, QInputDialog
)

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


# ----------------------------------------------------------------------
#  Reglage de l'etat de l'onglet Profils
# ----------------------------------------------------------------------

def _set_profils_state(win, current=True, reference=True, courbure=False,
                       pts=False, deviation=False, flap=False,
                       dev_mode='vertical'):
    """Positionne les cases de l'onglet Profils dans un etat connu."""
    tab = win._tab_profils
    tab._chk_current.setChecked(current)
    tab._chk_reference.setChecked(reference)
    tab._chk_porc_current.setChecked(courbure)
    tab._chk_sample_pts.setChecked(pts)
    tab._chk_flap.setChecked(flap)
    tab.set_deviation_mode(dev_mode)
    tab._chk_deviation.setChecked(deviation)


def capture_main_window(app, win):
    """Capture la fenetre principale au demarrage (etat par defaut riche)."""
    print("[1] Fenetre principale au demarrage")
    win._tabs.setCurrentWidget(win._tab_profils)
    win._tab_profils.zoom_fit()
    pump(app, 12)
    save(win, '01_main_window.png')


def capture_tab_profils(app, win):
    """Capture l'onglet Profils en differents etats."""
    tab = win._tab_profils
    win._tabs.setCurrentWidget(tab)

    print("[2] Onglet Profils - etat par defaut")
    tab.zoom_fit()
    pump(app, 10)
    save(win, '02_tab_profils_default.png')

    print("[3] Onglet Profils - mode Spline (points de controle)")
    _set_profils_state(win, current=True, reference=False)
    tab.zoom_fit()
    pump(app, 10)
    save(win, '03_tab_profils_spline.png')

    print("[4] Onglet Profils - courbure")
    _set_profils_state(win, current=True, reference=False, courbure=True)
    tab.zoom_fit()
    pump(app, 10)
    save(win, '04_tab_profils_courbure.png')

    print("[5] Onglet Profils - points echantillonnes")
    _set_profils_state(win, current=True, reference=False, pts=True)
    tab.zoom_fit()
    pump(app, 10)
    save(win, '05_tab_profils_pts.png')

    # Pour la deviation, utiliser une reference contrastee (NACA 0012)
    # afin que l'ecart soit nettement visible et illustratif.
    tab.load_profil_from_naca('0012', 'reference', n_points=150)

    print("[6] Onglet Profils - deviation (mode vertical)")
    _set_profils_state(win, current=True, reference=True,
                       deviation=True, dev_mode='vertical')
    tab.zoom_fit()
    pump(app, 10)
    save(win, '06_tab_profils_deviation.png')

    print("[14] Onglet Profils - deviation (mode normal)")
    _set_profils_state(win, current=True, reference=True,
                       deviation=True, dev_mode='normal')
    tab.zoom_fit()
    pump(app, 10)
    save(win, '14_deviation_normale.png')

    # Restaurer la reference NACA 2412 par defaut
    tab.load_profil_from_naca('2412', 'reference', n_points=150)

    print("[15] Onglet Profils - profil avec volet (flap)")
    _set_profils_state(win, current=True, reference=False, flap=True)
    tab.zoom_fit()
    pump(app, 10)
    save(win, '15_tab_profils_flap.png')

    # Reset etat par defaut
    _set_profils_state(win, current=True, reference=True,
                       courbure=True, deviation=True, flap=True)


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
    spn_max_seg.setValue(1)
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

    dlg.setMinimumWidth(360)
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


def capture_tab_results_empty(app, win):
    """Capture l'onglet Resultats (vide)."""
    print("[10] Onglet Resultats - vide")
    win._tabs.setCurrentWidget(win._tab_results)
    pump(app, 10)
    save(win, '10_tab_results_empty.png')


def capture_naca_dialog(app, win):
    """Capture le dialogue de saisie des indices NACA."""
    print("[16] Dialogue Profil NACA")
    dlg = QInputDialog(win)
    dlg.setInputMode(QInputDialog.TextInput)
    dlg.setWindowTitle(u"Profil NACA — courant")
    dlg.setLabelText(
        u"Indices NACA (4 ou 5 chiffres, ex. 2412 ou 23012) :")
    dlg.setTextValue("2412")
    dlg.resize(440, 130)
    dlg.show()
    pump(app, 10)
    save(dlg, '16_dialog_naca.png')
    dlg.close()


def capture_diagnostic_dialog(app, win):
    """Capture le dialogue de diagnostic XFoil."""
    print("[17] Dialogue Diagnostic XFoil")
    from gui.diagnostic_dialog import DiagnosticDialog

    wd = os.path.join(tempfile.gettempdir(), '_diag_demo')
    os.makedirs(wd, exist_ok=True)
    with open(os.path.join(wd, 'diagnostic.log'), 'w',
              encoding='utf-8') as f:
        f.write(
            "10:42:18 INFO    model.simulation: Simulation 'NACA 2412'"
            " avec xfoil (Re=1000000.0)\n"
            "10:42:18 INFO    model.pipeline: === Demarrage pipeline"
            " foil2d [xfoil] ===\n"
            "10:42:18 INFO    model.pipeline: --- Etape 1/3 :"
            " Preprocessing ---\n"
            "10:42:18 INFO    model.pipeline:   2 fichiers generes\n"
            "10:42:18 INFO    model.pipeline: --- Etape 2/3 :"
            " Simulation ---\n"
            "10:42:18 INFO    model.xfoil_simulator: XFoil termine"
            " avec succes\n"
            "10:42:18 INFO    model.pipeline: --- Etape 3/3 :"
            " Postprocessing ---\n"
            "10:42:18 INFO    model.xfoil_postprocessor: Resultats"
            " parses : 2 polaires, 27 distributions Cp\n"
            "10:42:18 INFO    model.simulation: Bilan : 27 point(s)"
            " converge(s) sur 2 Reynolds\n"
            "10:42:18 INFO    model.simulation: Simulation terminee :"
            " SimulationResults(2 Re, 27 pts converges, 0 warnings)\n")
    open(os.path.join(wd, 'xfoil_alpha.cmd'), 'w').write(
        "LOAD profil.dat\nOPER\nVISC 1000000\nASEQ -5 15 0.5\nQUIT\n")
    open(os.path.join(wd, 'polar_Re1000000.dat'), 'w').write(
        "XFoil polar\n  alpha   CL ...\n")

    dlg = DiagnosticDialog(wd, title=u"courant", parent=win)
    dlg.show()
    pump(app, 12)
    save(dlg, '17_dialog_diagnostic.png')
    dlg.close()


def capture_results_populated(app, win):
    """Lance une petite simulation et capture l'onglet Resultats peuple."""
    print("[18] Onglet Resultats - peuple (simulation reelle)")
    from model.simulation import Simulation

    params = {
        'RE_LIST': [1000000.0], 'RE': 1000000.0,
        'ALPHA_MIN': -3.0, 'ALPHA_MAX': 12.0, 'ALPHA_STEP': 1.0,
        'VISCOUS': True, 'NCRIT': 9.0, 'XTR_TOP': 0.2, 'XTR_BOT': 0.2,
        'REPANEL': True, 'NPANEL': 200, 'TIMEOUT': 90, 'ITER': 150,
        'USE_ASEQ': True, 'INIT_RETRY': False}

    tab = win._tab_profils
    jobs = [
        ('current', tab.profil_current, True),
        ('reference', tab.profil_reference, True),
        ('flap', tab.profil_flap_normalized(), False),
    ]
    results = {}
    for role, prof, norm in jobs:
        if prof is None:
            continue
        try:
            sim = Simulation(prof, params=params, normalize=norm)
            results[role] = sim.run()
            print(u"    %s : %d pts" % (role, results[role].n_converged))
        except Exception as e:
            print(u"    %s : echec (%s)" % (role, e))

    if not results:
        print("    (aucun resultat : XFoil indisponible ?) - ignore")
        return

    win._tab_results.set_results(results)
    # Configurer les 4 cellules pour des analyses representatives
    analyses = ['CL(alpha)', 'CL(CD)', 'Finesse(CL)', 'CM(CL)']
    for cell, name in zip(win._tab_results._cells, analyses):
        cell.set_analysis(name)
    win._tabs.setCurrentWidget(win._tab_results)
    pump(app, 15)
    save(win, '18_tab_results.png')


def _make_calque_asset(path):
    """Genere une image de calque synthetique (silhouette de profil)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

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
    win._tabs.setCurrentWidget(tab)
    _set_profils_state(win, current=True, reference=False)

    calque = os.path.join(tempfile.gettempdir(), '_calque_demo.png')
    _make_calque_asset(calque)
    tab.load_background_image(calque)
    tab._chk_image.setChecked(True)
    tab.zoom_fit()
    pump(app, 12)
    save(win, '12_tab_profils_calque.png')

    tab.clear_background_image()
    _set_profils_state(win, current=True, reference=True,
                       courbure=True, deviation=True, flap=True)
    try:
        os.remove(calque)
    except OSError:
        pass


def generate_ba_tangent_figure():
    """Genere la figure illustrant la contrainte de tangente au BA."""
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
        ax.plot(pts[:, 0], pts[:, 1], '-', color=blue, lw=2.0)
        ax.plot(cp[:, 0], cp[:, 1], '--', color=grey, lw=0.9)
        ax.plot(cp[1:-1, 0], cp[1:-1, 1], 's', color=blue, ms=7)
        ax.plot(cp[0, 0], cp[0, 1], 'o', color=blue, ms=6)
        ax.annotate('BA (P0)', xy=cp[0], xytext=(20, -28),
                    textcoords='offset points', fontsize=10)
        ax.annotate('P1', xy=cp[1], xytext=(14, 6),
                    textcoords='offset points', fontsize=10)
        ax.add_patch(Circle(cp[1], radius=22, fill=False,
                            edgecolor='#d62728', lw=2.0))
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


def generate_flap_schema_figure():
    """Schema geometrique du braquage de volet (cercle d'articulation)."""
    print("[19] Figure - schema geometrique du volet")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle
    from model.flap import hinge_circle

    prof = ProfilSpline.from_naca('2412', n_points=300)
    prof.normalize()
    ext = np.asarray(prof.extrados, dtype=float)
    intr = np.asarray(prof.intrados, dtype=float)
    chord = prof.chord
    xf = ext[0, 0] + 0.70 * chord
    C, r, ef, if_ = hinge_circle(ext, intr, xf)

    fig = Figure(figsize=(10, 4.2), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    pts = prof.points
    ax.plot(pts[:, 0], pts[:, 1], '-', color='#1f77b4', lw=1.6,
            label=u'profil')
    ax.add_patch(Circle(C, radius=r, fill=False, edgecolor='#2ca02c',
                        lw=1.8))
    ax.plot([C[0]], [C[1]], 'o', color='#d62728', ms=7)
    ax.annotate(u"axe d'articulation\n(centre du cercle inscrit)",
                xy=(C[0], C[1]), xytext=(120, 150),
                textcoords='offset points', fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#d62728'))
    ax.plot([ef[0]], [ef[1]], 's', color='#2ca02c', ms=7)
    ax.plot([if_[0]], [if_[1]], 's', color='#2ca02c', ms=7)
    ax.annotate(u"tangence extrados", xy=(ef[0], ef[1]),
                xytext=(-30, 70), textcoords='offset points', fontsize=9,
                ha='center',
                arrowprops=dict(arrowstyle='->', color='#2ca02c'))
    ax.annotate(u"tangence intrados", xy=(if_[0], if_[1]),
                xytext=(-30, -70), textcoords='offset points', fontsize=9,
                ha='center',
                arrowprops=dict(arrowstyle='->', color='#2ca02c'))
    ax.axvline(xf, color='#999999', ls=':', lw=1.0)
    ax.annotate(u"$X_f$ = 70 % de corde", xy=(xf, r + C[1]),
                xytext=(xf, 250), fontsize=9, ha='center',
                color='#555555',
                arrowprops=dict(arrowstyle='->', color='#999999'))
    ax.set_aspect('equal')
    ax.set_xlim(-60, 1060)
    ax.set_ylim(-230, 320)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.grid(True, alpha=0.3)
    ax.set_title(
        u"Articulation du volet : cercle inscrit tangent aux deux "
        u"surfaces,\ncentre a l'abscisse $X_f$", fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, '19_flap_schema.png')
    fig.savefig(path, dpi=110)
    print(u"  -> 19_flap_schema.png")


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
    # Echelle de courbure lisible pour les illustrations (le defaut
    # 200 fait sortir les quills du cadre pour une corde de 1000 mm).
    win._tab_profils._canvas._set_porcupine_scale(10.0)
    pump(app, 20)

    capture_main_window(app, win)
    capture_tab_profils(app, win)
    capture_calque(app, win)
    capture_convert_dialog(app, win)
    capture_courbure_warning(app, win)
    capture_naca_dialog(app, win)
    capture_diagnostic_dialog(app, win)
    capture_tab_xfoil(app, win)
    capture_tab_results_empty(app, win)
    capture_results_populated(app, win)

    generate_ba_tangent_figure()
    generate_flap_schema_figure()

    print("=" * 60)
    print("Termine.")
    print("=" * 60)

    win.close()


if __name__ == '__main__':
    main()
