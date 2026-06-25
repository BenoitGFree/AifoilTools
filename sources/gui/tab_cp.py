#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Onglet « Cp / Couche limite » : vue detaillee facon XFoil.

Pour un calcul precis (un profil parmi courant/reference/volet, un
Reynolds et une incidence), affiche dans une fenetre unique :
- la distribution de pression -Cp(x), separee extrados / intrados ;
- le profil redessine en dessous ;
- les frontieres de couche limite (surface + delta*) extrados / intrados.

Convention de couleurs (coherente avec l'ecran XFoil d'origine) :
extrados en jaune, intrados en bleu ; le contour du profil en gris.
"""

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
)

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)

from .i18n import tr as _
from .result_cell import (
    _cp_select, _coeffs_at, _bl_at, _format_re)


# Couleurs facon XFoil (PltLib) : fond noir, extrados jaune, intrados
# cyan, profil et grille en blanc.
COLOR_BG = '#000000'         # fond noir
COLOR_FG = '#FFFFFF'         # axes, grille, profil, texte
COLOR_EXTRADOS = '#FFFF00'   # jaune (surface superieure)
COLOR_INTRADOS = '#00E5FF'   # cyan (surface inferieure)
COLOR_PROFIL = '#FFFFFF'     # contour du profil (blanc)

# Exageration verticale du profil dans la bande basse, pour rendre les
# limites de couche limite (delta*) nettement visibles, comme XFoil.
FOIL_YSCALE = 2.5

# Libelles des roles
_ROLE_LABELS = {
    'current': u'Courant',
    'reference': u'Référence',
    'flap': u'Volet',
}


def _split_le(x):
    u"""Indice du bord d'attaque (abscisse minimale) d'un contour Selig.

    :param x: tableau d'abscisses ordonnees BF -> extrados -> BA ->
        intrados -> BF
    :returns: indice du bord d'attaque
    :rtype: int
    """
    return int(np.argmin(x))


def _airfoil_surfaces(x, y, dstar=None, s=None, eps=1e-6):
    u"""Separe extrados / intrados d'un profil (sans sillage).

    Methode TOPOLOGIQUE robuste (valable profil symetrique ou cambre) :

    1. retirer le sillage (x > 1) ;
    2. ordonner les points le long du contour par l'abscisse curviligne
       ``s`` (le DUMP XFoil decrit un contour continu BF -> extrados ->
       BA -> intrados -> BF) ;
    3. couper le contour au bord d'attaque (abscisse x minimale) : un arc
       = extrados, l'autre = intrados ;
    4. l'extrados est l'arc d'ordonnee moyenne la plus haute.

    Pas d'estimation de cambrure (qui derapait au BF des profils cambres
    et y melangeait extrados/intrados). Repli sans ``s`` : tri par x puis
    decoupe au BA (moins fiable au BF epais).

    :returns: ((xe, ye, de), (xi, yi, di)) ordonnes le long du contour,
        ou None. de / di valent None si dstar n'est pas fourni.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = x <= 1.0 + eps                 # retirer le sillage (x > 1)
    x, y = x[mask], y[mask]
    d = (np.asarray(dstar, dtype=float)[mask]
         if dstar is not None else None)
    if len(x) < 4:
        return None

    # Ordre le long du contour
    if s is not None:
        order = np.argsort(np.asarray(s, dtype=float)[mask], kind='stable')
    else:
        order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    if d is not None:
        d = d[order]

    # Coupe au bord d'attaque (x minimal) : deux arcs du contour
    le = int(np.argmin(x))
    arc_a = np.arange(0, le + 1)
    arc_b = np.arange(le, len(x))
    if len(arc_a) < 2 or len(arc_b) < 2:
        return None

    def _pack(idx):
        ds = d[idx] if d is not None else None
        return x[idx], y[idx], ds

    # L'extrados est l'arc dont l'ordonnee moyenne est la plus haute
    if np.mean(y[arc_a]) >= np.mean(y[arc_b]):
        return _pack(arc_a), _pack(arc_b)
    return _pack(arc_b), _pack(arc_a)


def _bl_envelope_side(x, y, dstar, want_up):
    u"""Frontiere de couche limite d'un cote : surface + delta* (normale).

    La normale unitaire est forcee vers l'exterieur : vers le haut pour
    l'extrados (want_up=True), vers le bas pour l'intrados. Le signe est
    decide globalement sur le cote (la normale ne change pas de sens le
    long d'une surface allant du BA au BF).

    :returns: (x_bl, y_bl)
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx ** 2 + dy ** 2)
    norm = np.where(norm < 1e-15, 1.0, norm)
    nx = dy / norm
    ny = -dx / norm
    # Orienter la normale vers l'exterieur du profil
    if (np.mean(ny) > 0) != want_up:
        nx, ny = -nx, -ny
    return x + nx * dstar, y + ny * dstar


class TabCp(QWidget):
    u"""Onglet de visualisation Cp + couche limite pour un calcul precis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = {}     # {'current': SimulationResults, ...}
        self._build_ui()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Barre de selection ---
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        ctrl.addWidget(QLabel(_("Profil :")))
        self._combo_role = QComboBox()
        self._combo_role.setToolTip(
            _(u"Profil a visualiser : courant (bleu), reference (rouge)"
              u" ou volet (vert)."))
        self._combo_role.currentIndexChanged.connect(
            self._on_role_changed)
        ctrl.addWidget(self._combo_role)

        ctrl.addWidget(QLabel(_("Re :")))
        self._combo_re = QComboBox()
        self._combo_re.setMinimumWidth(90)
        self._combo_re.setToolTip(_(u"Nombre de Reynolds du calcul."))
        self._combo_re.currentIndexChanged.connect(self._on_re_changed)
        ctrl.addWidget(self._combo_re)

        ctrl.addWidget(QLabel(_("α :")))
        self._combo_alpha = QComboBox()
        self._combo_alpha.setMinimumWidth(80)
        self._combo_alpha.setToolTip(_(u"Incidence (degres) du calcul."))
        self._combo_alpha.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._combo_alpha)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # --- Canvas (fond noir facon XFoil) ---
        # Deux reperes partageant l'axe x : la pression (Cp) en haut, le
        # profil avec sa couche limite en bas, ce dernier a aspect EGAL
        # pour une geometrie correcte (delta* perpendiculaire, a l'echelle
        # de corde).
        self._fig = Figure(figsize=(6, 6), dpi=100, facecolor=COLOR_BG)
        gs = self._fig.add_gridspec(
            2, 1, height_ratios=[3.0, 1.4], hspace=0.06,
            left=0.10, right=0.97, top=0.95, bottom=0.08)
        self._ax_cp = self._fig.add_subplot(gs[0])
        self._ax_foil = self._fig.add_subplot(gs[1], sharex=self._ax_cp)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, stretch=1)

        # Message initial
        self._show_empty(_("Aucun resultat. Lancez une simulation."))

    def _style_axes(self, ax):
        u"""Applique le style XFoil (fond noir, traits blancs) a un axe."""
        ax.set_facecolor(COLOR_BG)
        for spine in ax.spines.values():
            spine.set_color(COLOR_FG)
        ax.tick_params(colors=COLOR_FG, labelsize=8)
        ax.grid(True, color=COLOR_FG, alpha=0.30,
                linestyle=':', linewidth=0.6)

    def _show_empty(self, message):
        u"""Affiche un message au centre du canvas vide."""
        self._ax_foil.set_visible(False)
        self._ax_cp.clear()
        self._ax_cp.set_facecolor(COLOR_BG)
        self._ax_cp.text(0.5, 0.5, message, ha='center', va='center',
                         transform=self._ax_cp.transAxes, color='#888888',
                         fontsize=10)
        self._ax_cp.set_xticks([])
        self._ax_cp.set_yticks([])
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def set_results(self, results):
        u"""Definit les resultats de simulation et rafraichit l'affichage.

        :param results: dict {'current': SimulationResults, ...}
        """
        self._results = results or {}
        self._populate_role_combo()
        self._replot()

    def update_results(self, new_results):
        u"""Fusionne de nouveaux resultats (relance partielle d'un profil).

        :param new_results: dict {'current': SimulationResults, ...}
        """
        if not new_results:
            return
        self._results.update(new_results)
        self._populate_role_combo()
        self._replot()

    # ------------------------------------------------------------------
    # Combos en cascade : profil -> Re -> alpha
    # ------------------------------------------------------------------

    def _available_roles(self):
        u"""Roles ayant des donnees Cp ou couche limite."""
        roles = []
        for role, sim in self._results.items():
            if sim is not None and (sim.has_cp or sim.has_bl):
                roles.append(role)
        # Ordre stable : courant, reference, volet
        order = {'current': 0, 'reference': 1, 'flap': 2}
        return sorted(roles, key=lambda r: order.get(r, 9))

    def _populate_role_combo(self):
        self._combo_role.blockSignals(True)
        old = self._combo_role.currentData()
        self._combo_role.clear()
        for role in self._available_roles():
            self._combo_role.addItem(
                _(_ROLE_LABELS.get(role, role)), role)
        idx = self._combo_role.findData(old)
        if idx >= 0:
            self._combo_role.setCurrentIndex(idx)
        self._combo_role.blockSignals(False)
        self._populate_re_combo()

    def _current_sim(self):
        role = self._combo_role.currentData()
        if role is None:
            return None
        return self._results.get(role)

    def _re_values(self, sim):
        u"""Liste triee des Reynolds disponibles (Cp ou BL)."""
        re_set = set()
        if sim is None:
            return []
        if sim.has_cp:
            re_set.update(k for k in sim.cp if isinstance(k, float))
        if sim.has_bl:
            re_set.update(k for k in sim.bl if isinstance(k, float))
        return sorted(re_set)

    def _populate_re_combo(self):
        self._combo_re.blockSignals(True)
        old = self._combo_re.currentData()
        self._combo_re.clear()
        sim = self._current_sim()
        for re_val in self._re_values(sim):
            self._combo_re.addItem(_format_re(re_val), re_val)
        idx = self._combo_re.findData(old)
        if idx >= 0:
            self._combo_re.setCurrentIndex(idx)
        self._combo_re.blockSignals(False)
        self._populate_alpha_combo()

    def _alpha_values(self, sim, re_val):
        u"""Liste triee des incidences disponibles pour (sim, Re)."""
        a_set = set()
        if sim is None or re_val is None:
            return []
        if sim.has_cp and re_val in sim.cp:
            a_set.update(float(a) for a in sim.cp[re_val])
        if sim.has_bl and re_val in sim.bl:
            a_set.update(float(a) for a in sim.bl[re_val])
        return sorted(a_set)

    def _populate_alpha_combo(self):
        self._combo_alpha.blockSignals(True)
        old = self._combo_alpha.currentData()
        self._combo_alpha.clear()
        sim = self._current_sim()
        re_val = self._combo_re.currentData()
        for a in self._alpha_values(sim, re_val):
            self._combo_alpha.addItem(u'%.2f°' % a, a)
        idx = self._combo_alpha.findData(old)
        if idx >= 0:
            self._combo_alpha.setCurrentIndex(idx)
        self._combo_alpha.blockSignals(False)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_role_changed(self, _index=None):
        self._populate_re_combo()
        self._replot()

    def _on_re_changed(self, _index=None):
        self._populate_alpha_combo()
        self._replot()

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    def _replot(self, *args):
        u"""Retrace la vue Cp (haut) + profil/couche limite (bas)."""
        if not self._available_roles():
            self._show_empty(_("Aucun resultat. Lancez une simulation."))
            return

        sim = self._current_sim()
        re_val = self._combo_re.currentData()
        alpha = self._combo_alpha.currentData()
        role = self._combo_role.currentData()
        if sim is None or re_val is None:
            self._show_empty(_("Calcul indisponible."))
            return

        self._ax_foil.set_visible(True)
        self._ax_cp.clear()
        self._ax_foil.clear()
        self._style_axes(self._ax_cp)
        self._style_axes(self._ax_foil)

        cp_drawn = self._plot_pressure(sim, re_val, alpha)
        foil_drawn = self._plot_airfoil_bl(sim, re_val, alpha)

        if not cp_drawn and not foil_drawn:
            self._show_empty(_("Pas de Cp ni de couche limite pour ce"
                               " calcul."))
            return

        # --- Encart de coefficients (sur le repere Cp) ---
        coeffs = _coeffs_at(sim, re_val, alpha)
        self._annotate(role, re_val, alpha, coeffs)

        # Titre
        title = _("Cp + couche limite")
        title += '  %s' % _format_re(re_val)
        if alpha is not None:
            title += u'  α=%.2f°' % alpha
        role_lbl = _(_ROLE_LABELS.get(role, role or ''))
        self._ax_cp.set_title('%s — %s' % (role_lbl, title),
                              fontsize=10, color=COLOR_FG)

        self._canvas.draw_idle()

    def _plot_pressure(self, sim, re_val, alpha):
        u"""Trace Cp(x) (axe inverse) : visqueux + non visqueux."""
        ax = self._ax_cp
        ax.set_ylabel('Cp', fontsize=9, color=COLOR_FG)
        drawn = False

        # Cp visqueux (extrados / intrados)
        sel = _cp_select(sim, re_val, alpha)
        if sel is not None:
            _re, _a, cp_data = sel
            x = cp_data[:, 0]
            cp = cp_data[:, -1]
            le = _split_le(x)
            ax.plot(x[:le + 1], cp[:le + 1], color=COLOR_EXTRADOS,
                    linewidth=1.3, label=_("Extrados"))
            ax.plot(x[le:], cp[le:], color=COLOR_INTRADOS,
                    linewidth=1.3, label=_("Intrados"))
            drawn = True

        # Cp non visqueux (pointille, une seule courbe continue)
        cpi = sim.get_cpi(alpha) if sim.has_cpi else None
        if cpi is not None and len(cpi) > 0:
            ax.plot(cpi[:, 0], cpi[:, -1], color=COLOR_FG,
                    linewidth=0.8, linestyle='--', alpha=0.7,
                    label=_("Non visqueux"))
            drawn = True

        if not drawn:
            return False

        # Convention XFoil : Cp croissant vers le bas (succion en haut)
        ax.invert_yaxis()
        ax.axhline(0.0, color=COLOR_FG, linewidth=0.6, alpha=0.5)
        ax.tick_params(labelbottom=False)
        leg = ax.legend(fontsize=7, loc='lower right', framealpha=0.0,
                        labelcolor=COLOR_FG)
        if leg is not None:
            leg.get_frame().set_edgecolor('none')
        return True

    def _plot_airfoil_bl(self, sim, re_val, alpha):
        u"""Trace le profil et les limites de couche limite (delta*).

        Le repere est a aspect EGAL : le profil est a la vraie echelle de
        corde et les frontieres delta* sont normales a la surface.

        Le classement extrados/intrados est geometrique (voir
        _airfoil_surfaces) ; le tri se fait par abscisse curviligne s,
        et les points de sillage (x > 1) sont exclus.

        :returns: True si quelque chose a ete trace
        """
        ax = self._ax_foil
        ax.set_xlabel('x/c', fontsize=9, color=COLOR_FG)

        bl = _bl_at(sim.bl.get(re_val), alpha) if sim.has_bl else None

        if bl is not None and 'x' in bl and 'y' in bl:
            xf = np.asarray(bl['x'], dtype=float)
            yf = np.asarray(bl['y'], dtype=float)
            dstar = (np.asarray(bl['Dstar'], dtype=float)
                     if 'Dstar' in bl else None)
            sf = np.asarray(bl['s'], dtype=float) if 's' in bl else None
            self._draw_dump_airfoil(ax, xf, yf, dstar, sf)
        else:
            # Pas de couche limite : geometrie (ordre Selig) depuis le Cp
            sel = _cp_select(sim, re_val, alpha)
            if sel is None or sel[2].shape[1] < 3:
                ax.set_visible(False)
                return False
            cp_data = sel[2]
            ax.plot(cp_data[:, 0], cp_data[:, 1], color=COLOR_PROFIL,
                    linewidth=1.2, label=_("Profil"))

        # Aspect egal : geometrie correcte (profil a l'echelle de corde)
        ax.set_aspect('equal', adjustable='box')
        ax.set_yticks([])
        return True

    def _draw_dump_airfoil(self, ax, xf, yf, dstar, sf=None):
        u"""Trace profil + delta* a partir des donnees DUMP.

        Le classement extrados/intrados est geometrique (voir
        _airfoil_surfaces), donc independant de l'ordre des points et du
        sillage. Tri le long de la surface par abscisse curviligne sf.
        """
        res = _airfoil_surfaces(xf, yf, dstar, sf)
        if res is None:
            ax.plot(xf, yf, color=COLOR_PROFIL, linewidth=1.2,
                    label=_("Profil"))
            return
        (xe, ye, de), (xi, yi, di) = res
        specs = [
            (xe, ye, de, COLOR_EXTRADOS, True, _("δ* extrados")),
            (xi, yi, di, COLOR_INTRADOS, False, _("δ* intrados")),
        ]
        profil_label = _("Profil")
        for xs, ys, ds, color, want_up, dstar_label in specs:
            if len(xs) < 2:
                continue
            # Contour du profil (chaque cote trace separement)
            ax.plot(xs, ys, color=COLOR_PROFIL, linewidth=1.2,
                    label=profil_label)
            profil_label = None  # une seule entree de legende
            # Limite de couche limite (surface + delta* normal)
            if ds is not None and len(ds) == len(xs):
                xbl, ybl = _bl_envelope_side(xs, ys, ds, want_up)
                ax.plot(xbl, ybl, color=color, linewidth=1.0,
                        label=dstar_label)

    def _annotate(self, role, re_val, alpha, coeffs):
        u"""Encart de coefficients (coin haut-droit, façon XFoil)."""
        def _fmt(v, fmt):
            return (fmt % v) if v is not None else u'—'
        lines = [
            _(_ROLE_LABELS.get(role, role or '')),
            u'Re  = %s' % _format_re(re_val).replace('Re=', ''),
            u'α   = %.2f°' % (alpha if alpha is not None else 0.0),
            u'CL  = %s' % _fmt(coeffs['CL'], '%.4f'),
            u'CM  = %s' % _fmt(coeffs['CM'], '%.4f'),
            u'CD  = %s' % _fmt(coeffs['CD'], '%.5f'),
            u'L/D = %s' % _fmt(coeffs['LD'], '%.2f'),
        ]
        self._ax_cp.text(
            0.985, 0.97, '\n'.join(lines),
            transform=self._ax_cp.transAxes, ha='right', va='top',
            fontsize=8, family='monospace', color=COLOR_FG)
