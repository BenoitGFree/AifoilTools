#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Cellule d'analyse individuelle pour l'onglet Resultats.

Chaque cellule contient :
- Un QComboBox pour choisir le type d'analyse
- Un canvas matplotlib interactif (zoom, pan, curseur)
- Des checkboxes pour afficher courant / reference
"""

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QLabel
)
from PySide6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


# Couleurs (coherentes avec profil_canvas et tab_results)
COLOR_CURRENT = '#1f77b4'      # bleu
COLOR_REFERENCE = '#d62728'    # rouge

# Styles de ligne par Reynolds (cyclique)
RE_LINESTYLES = ['-', '--', '-.', ':']


# ======================================================================
# Fonctions de trace (plot_fn)
# ======================================================================
# Signature : plot_fn(ax, sim_results, color, label_base)
# Trace les courbes pour un role (courant ou reference) pour tous les Re.

def _plot_cl_alpha(ax, sim_results, color, label_base):
    """CL en fonction de alpha."""
    _plot_polar_xy(ax, sim_results, color, label_base, 'alpha', 'CL')


def _plot_cd_alpha(ax, sim_results, color, label_base):
    """CD en fonction de alpha."""
    _plot_polar_xy(ax, sim_results, color, label_base, 'alpha', 'CD')


def _plot_finesse_alpha(ax, sim_results, color, label_base):
    """Finesse (CL/CD) en fonction de alpha."""
    re_list = sim_results.re_list
    single = (len(re_list) == 1)
    for i, re_val in enumerate(re_list):
        polar = sim_results.get_polar(re_val)
        if polar is None:
            continue
        ls = '-' if single else RE_LINESTYLES[i % len(RE_LINESTYLES)]
        lbl = label_base if single else '%s %s' % (label_base, _format_re(re_val))
        alpha = polar['alpha']
        cl = polar['CL']
        cd = polar['CD']
        finesse = np.where(cd > 1e-8, cl / cd, 0.0)
        ax.plot(alpha, finesse, color=color, linestyle=ls, label=lbl)


def _plot_cl_cd(ax, sim_results, color, label_base):
    """CL en fonction de CD (trainee polaire)."""
    _plot_polar_xy(ax, sim_results, color, label_base, 'CD', 'CL')


def _plot_cm_alpha(ax, sim_results, color, label_base):
    """CM en fonction de alpha."""
    _plot_polar_xy(ax, sim_results, color, label_base, 'alpha', 'CM')


def _plot_xtr_alpha(ax, sim_results, color, label_base):
    """Transition (Top et Bot) en fonction de alpha."""
    re_list = sim_results.re_list
    single = (len(re_list) == 1)
    for i, re_val in enumerate(re_list):
        polar = sim_results.get_polar(re_val)
        if polar is None:
            continue
        ls = '-' if single else RE_LINESTYLES[i % len(RE_LINESTYLES)]
        re_tag = '' if single else ' %s' % _format_re(re_val)
        alpha = polar['alpha']
        ax.plot(alpha, polar['Top_Xtr'], color=color, linestyle=ls,
                label='%s Top%s' % (label_base, re_tag))
        ax.plot(alpha, polar['Bot_Xtr'], color=color, linestyle=ls,
                alpha=0.6, label='%s Bot%s' % (label_base, re_tag))


def _plot_cdp_alpha(ax, sim_results, color, label_base):
    """CDp (trainee de pression) en fonction de alpha."""
    _plot_polar_xy(ax, sim_results, color, label_base, 'alpha', 'CDp')


def _plot_polar_xy(ax, sim_results, color, label_base, x_key, y_key):
    """Trace generique x_key vs y_key depuis les polaires."""
    re_list = sim_results.re_list
    single = (len(re_list) == 1)
    for i, re_val in enumerate(re_list):
        polar = sim_results.get_polar(re_val)
        if polar is None:
            continue
        ls = '-' if single else RE_LINESTYLES[i % len(RE_LINESTYLES)]
        lbl = label_base if single else '%s %s' % (label_base, _format_re(re_val))
        ax.plot(polar[x_key], polar[y_key],
                color=color, linestyle=ls, label=lbl)


def _plot_cp_x(ax, sim_results, color, label_base, re_selected=None):
    u"""-Cp en fonction de x/c (distribution de pression).

    Trace pour le Re selectionne, toutes les incidences.
    Convention aero : on trace -Cp (succion vers le haut).

    :param re_selected: Re a tracer (None = premier disponible)
    """
    if not sim_results.has_cp:
        return

    cp_dict = sim_results.cp
    re_vals = sorted(k for k in cp_dict if isinstance(k, float))
    if not re_vals:
        return

    re_val = re_selected if re_selected in cp_dict else re_vals[0]
    alphas = sorted(cp_dict[re_val].keys())
    if not alphas:
        return

    n_alpha = len(alphas)
    for i, alpha in enumerate(alphas):
        cp_data = cp_dict[re_val].get(alpha)
        if cp_data is None or len(cp_data) == 0:
            continue
        ls = '-' if n_alpha == 1 else RE_LINESTYLES[
            i % len(RE_LINESTYLES)]
        lbl = u'%s \u03b1=%.1f\u00b0' % (label_base, alpha)
        ax.plot(cp_data[:, 0], -cp_data[:, 1],
                color=color, linestyle=ls, linewidth=0.8,
                label=lbl)


def _plot_bl_xy(ax, sim_results, color, label_base, y_key, ylabel):
    u"""Trace generique d'une variable BL en fonction de s."""
    bl_dict = sim_results.bl
    re_vals = sorted(k for k in bl_dict
                     if isinstance(k, float) and bl_dict[k] is not None)
    if not re_vals:
        return
    single = (len(re_vals) == 1)
    for i, re_val in enumerate(re_vals):
        bl = bl_dict[re_val]
        ls = '-' if single else RE_LINESTYLES[i % len(RE_LINESTYLES)]
        lbl = label_base if single else (
            '%s %s' % (label_base, _format_re(re_val)))
        ax.plot(bl['s'], bl[y_key],
                color=color, linestyle=ls, label=lbl)


def _plot_ue_s(ax, sim_results, color, label_base):
    u"""Vitesse de bord de couche limite en fonction de s."""
    _plot_bl_xy(ax, sim_results, color, label_base, 'Ue_Vinf', 'Ue/Vinf')


def _plot_dstar_s(ax, sim_results, color, label_base):
    u"""Epaisseur de deplacement en fonction de s."""
    _plot_bl_xy(ax, sim_results, color, label_base, 'Dstar', u'\u03b4*')


def _plot_theta_s(ax, sim_results, color, label_base):
    u"""Epaisseur de quantite de mouvement en fonction de s."""
    _plot_bl_xy(ax, sim_results, color, label_base, 'Theta', u'\u03b8')


def _plot_cf_s(ax, sim_results, color, label_base):
    u"""Coefficient de frottement en fonction de s."""
    _plot_bl_xy(ax, sim_results, color, label_base, 'Cf', 'Cf')


def _plot_h_s(ax, sim_results, color, label_base):
    u"""Facteur de forme en fonction de s."""
    _plot_bl_xy(ax, sim_results, color, label_base, 'H', 'H')


def _plot_bl_x(ax, sim_results, color, label_base, y_key):
    u"""Trace generique d'une variable BL en fonction de x."""
    bl_dict = sim_results.bl
    re_vals = sorted(k for k in bl_dict
                     if isinstance(k, float) and bl_dict[k] is not None)
    if not re_vals:
        return
    single = (len(re_vals) == 1)
    for i, re_val in enumerate(re_vals):
        bl = bl_dict[re_val]
        ls = '-' if single else RE_LINESTYLES[i % len(RE_LINESTYLES)]
        lbl = label_base if single else (
            '%s %s' % (label_base, _format_re(re_val)))
        ax.plot(bl['x'], bl[y_key],
                color=color, linestyle=ls, label=lbl)


def _plot_ue_x(ax, sim_results, color, label_base):
    u"""Vitesse de bord de couche limite en fonction de x."""
    _plot_bl_x(ax, sim_results, color, label_base, 'Ue_Vinf')


def _plot_dstar_x(ax, sim_results, color, label_base):
    u"""Epaisseur de deplacement en fonction de x."""
    _plot_bl_x(ax, sim_results, color, label_base, 'Dstar')


def _plot_theta_x(ax, sim_results, color, label_base):
    u"""Epaisseur de quantite de mouvement en fonction de x."""
    _plot_bl_x(ax, sim_results, color, label_base, 'Theta')


def _plot_cf_x(ax, sim_results, color, label_base):
    u"""Coefficient de frottement en fonction de x."""
    _plot_bl_x(ax, sim_results, color, label_base, 'Cf')


def _plot_h_x(ax, sim_results, color, label_base):
    u"""Facteur de forme en fonction de x."""
    _plot_bl_x(ax, sim_results, color, label_base, 'H')


def _format_re(re_val):
    u"""Formate un Reynolds pour la legende."""
    if re_val >= 1e6:
        return 'Re=%.0fM' % (re_val / 1e6)
    elif re_val >= 1e3:
        return 'Re=%.0fk' % (re_val / 1e3)
    else:
        return 'Re=%g' % re_val


# ======================================================================
# Registre des analyses disponibles
# ======================================================================

ANALYSIS_REGISTRY = [
    # (nom_affiche, plot_fn, xlabel, ylabel)
    ('CL(alpha)',      _plot_cl_alpha,      'alpha (deg)', 'CL'),
    ('CD(alpha)',      _plot_cd_alpha,      'alpha (deg)', 'CD'),
    ('Finesse(alpha)', _plot_finesse_alpha, 'alpha (deg)', 'CL/CD'),
    ('CL(CD)',         _plot_cl_cd,         'CD',          'CL'),
    ('-Cp(x)',         _plot_cp_x,          'x/c',         '-Cp'),
    ('CM(alpha)',      _plot_cm_alpha,      'alpha (deg)', 'CM'),
    ('Xtr(alpha)',     _plot_xtr_alpha,     'alpha (deg)', 'Xtr'),
    ('CDp(alpha)',     _plot_cdp_alpha,     'alpha (deg)', 'CDp'),
    ('Ue(s)',          _plot_ue_s,          's',           'Ue/Vinf'),
    (u'Dstar(s)',      _plot_dstar_s,       's',           u'\u03b4*'),
    (u'Theta(s)',      _plot_theta_s,       's',           u'\u03b8'),
    ('Cf(s)',          _plot_cf_s,          's',           'Cf'),
    ('H(s)',           _plot_h_s,           's',           'H'),
    ('Ue(x)',          _plot_ue_x,          'x',           'Ue/Vinf'),
    (u'Dstar(x)',      _plot_dstar_x,       'x',           u'\u03b4*'),
    (u'Theta(x)',      _plot_theta_x,       'x',           u'\u03b8'),
    ('Cf(x)',          _plot_cf_x,          'x',           'Cf'),
    ('H(x)',           _plot_h_x,           'x',           'H'),
]

# Noms pour acces rapide
ANALYSIS_NAMES = [a[0] for a in ANALYSIS_REGISTRY]

# Disposition par defaut (ordre des analyses dans une grille 2x2)
DEFAULT_ANALYSES = ['CL(alpha)', 'CD(alpha)', 'Finesse(alpha)', 'CL(CD)']


def _get_analysis(name):
    """Retourne (plot_fn, xlabel, ylabel) pour un nom d'analyse."""
    for a_name, plot_fn, xlabel, ylabel in ANALYSIS_REGISTRY:
        if a_name == name:
            return plot_fn, xlabel, ylabel
    return None, None, None


# ======================================================================
# Widget ResultCell
# ======================================================================

class ResultCell(QWidget):
    u"""Cellule d'analyse individuelle avec canvas interactif."""

    def __init__(self, analysis_name='CL(alpha)', parent=None):
        super().__init__(parent)
        self._results = {}          # {'current': SimulationResults, ...}
        self._show_current = True
        self._show_reference = True
        self._analysis_name = analysis_name

        # Pan state
        self._panning = False
        self._pan_start_xy = None

        # Mapping legende -> ligne pour toggle visibilite
        self._legend_map = {}  # {legend_line: original_line}

        self._build_ui()
        self._set_combo_value(analysis_name)

    def _build_ui(self):
        u"""Construit l'interface de la cellule."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # --- Barre de controle ---
        ctrl = QHBoxLayout()
        ctrl.setSpacing(4)

        self._combo = QComboBox()
        self._combo.addItems(ANALYSIS_NAMES)
        self._combo.currentTextChanged.connect(self._on_analysis_changed)
        ctrl.addWidget(self._combo)

        self._chk_current = QCheckBox("C")
        self._chk_current.setChecked(True)
        self._chk_current.setToolTip("Afficher le profil courant")
        self._chk_current.setStyleSheet("color: %s;" % COLOR_CURRENT)
        self._chk_current.stateChanged.connect(self._on_toggle_current)
        ctrl.addWidget(self._chk_current)

        self._chk_reference = QCheckBox("R")
        self._chk_reference.setChecked(True)
        self._chk_reference.setToolTip(
            u"Afficher le profil r\u00e9f\u00e9rence")
        self._chk_reference.setStyleSheet(
            "color: %s;" % COLOR_REFERENCE)
        self._chk_reference.stateChanged.connect(
            self._on_toggle_reference)
        ctrl.addWidget(self._chk_reference)

        # Combo Reynolds (visible pour -Cp(x))
        self._combo_re = QComboBox()
        self._combo_re.setToolTip("Reynolds")
        self._combo_re.setMinimumWidth(90)
        self._combo_re.currentTextChanged.connect(
            self._on_re_changed)
        self._combo_re.setVisible(False)
        ctrl.addWidget(self._combo_re)

        self._lbl_cursor = QLabel("")
        self._lbl_cursor.setStyleSheet("font-size: 9px; color: #666;")
        ctrl.addWidget(self._lbl_cursor)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # --- Canvas matplotlib ---
        self._fig = Figure(figsize=(4, 3), dpi=100)
        self._fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._ax.grid(True, alpha=0.3)
        layout.addWidget(self._canvas, stretch=1)

        # --- Connecter les evenements souris ---
        self._canvas.mpl_connect('button_press_event', self._on_press)
        self._canvas.mpl_connect('motion_notify_event', self._on_motion)
        self._canvas.mpl_connect('button_release_event', self._on_release)
        self._canvas.mpl_connect('scroll_event', self._on_scroll)
        self._canvas.mpl_connect('pick_event', self._on_pick_legend)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def set_results(self, results):
        u"""Definit les resultats de simulation.

        :param results: dict {'current': SimulationResults, ...}
        """
        self._results = results
        self._populate_re_combo()
        self._replot()

    def set_analysis(self, name):
        u"""Change le type d'analyse affiche.

        :param name: nom de l'analyse (ex: 'CL(alpha)')
        """
        self._analysis_name = name
        self._set_combo_value(name)
        self._replot()

    @property
    def analysis_name(self):
        """Retourne le nom de l'analyse courante."""
        return self._analysis_name

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    def _replot(self):
        u"""Retrace le graphique avec l'analyse et les resultats courants."""
        # Sauvegarder les labels masques avant le clear
        hidden_labels = set()
        for leg_line, orig_line in self._legend_map.items():
            if not orig_line.get_visible():
                hidden_labels.add(orig_line.get_label())

        self._ax.clear()
        self._ax.grid(True, alpha=0.3)

        plot_fn, xlabel, ylabel = _get_analysis(self._analysis_name)
        if plot_fn is None:
            self._canvas.draw_idle()
            return

        is_cp = (self._analysis_name == '-Cp(x)')
        is_bl = self._analysis_name in (
            'Ue(s)', 'Dstar(s)', 'Theta(s)', 'Cf(s)', 'H(s)',
            'Ue(x)', 'Dstar(x)', 'Theta(x)', 'Cf(x)', 'H(x)')
        for role, sim_results in self._results.items():
            if is_cp and not sim_results.has_cp:
                continue
            if is_bl and not sim_results.has_bl:
                continue
            if not is_cp and not is_bl and not sim_results.has_polars:
                continue
            if role == 'current' and not self._show_current:
                continue
            if role == 'reference' and not self._show_reference:
                continue
            color = COLOR_CURRENT if role == 'current' \
                else COLOR_REFERENCE
            label_base = "Courant" if role == 'current' \
                else u"R\u00e9f."
            if is_cp:
                plot_fn(self._ax, sim_results, color,
                        label_base, self._selected_re)
            else:
                plot_fn(self._ax, sim_results, color,
                        label_base)

        self._ax.set_xlabel(xlabel, fontsize=8)
        self._ax.set_ylabel(ylabel, fontsize=8)
        title = self._analysis_name
        if is_cp and self._selected_re is not None:
            title += '  %s' % _format_re(self._selected_re)
        self._ax.set_title(title, fontsize=9)
        self._ax.tick_params(labelsize=7)

        # Legende interactive (clic pour masquer/afficher une courbe)
        self._legend_map = {}
        handles, labels = self._ax.get_legend_handles_labels()
        if labels:
            legend = self._ax.legend(fontsize=6)
            for leg_line, orig_line in zip(legend.get_lines(), handles):
                leg_line.set_picker(5)  # tolerance en pixels
                self._legend_map[leg_line] = orig_line
                # Restaurer l'etat masque
                if orig_line.get_label() in hidden_labels:
                    orig_line.set_visible(False)
                    leg_line.set_alpha(0.2)

        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Slots controles
    # ------------------------------------------------------------------

    def _on_analysis_changed(self, name):
        u"""Appele quand le combo d'analyse change."""
        self._analysis_name = name
        self._combo_re.setVisible(name == '-Cp(x)')
        self._replot()

    def _on_re_changed(self, text):
        u"""Appele quand le combo Reynolds change."""
        self._replot()

    def _on_toggle_current(self, state):
        self._show_current = (state == Qt.Checked.value)
        self._replot()

    def _on_toggle_reference(self, state):
        self._show_reference = (state == Qt.Checked.value)
        self._replot()

    def _on_pick_legend(self, event):
        u"""Toggle la visibilite d'une courbe en cliquant sur la legende."""
        leg_line = event.artist
        if leg_line not in self._legend_map:
            return
        orig_line = self._legend_map[leg_line]
        visible = not orig_line.get_visible()
        orig_line.set_visible(visible)
        # Attenuer la ligne de legende si la courbe est masquee
        leg_line.set_alpha(1.0 if visible else 0.2)
        self._canvas.draw_idle()

    def _show_context_menu(self, event):
        u"""Menu contextuel : tout afficher / tout masquer."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtCore import QPoint

        if not self._legend_map:
            return

        menu = QMenu(self._canvas)
        act_show = menu.addAction("Tout afficher")
        act_hide = menu.addAction("Tout masquer")

        act_show.triggered.connect(self._on_show_all_curves)
        act_hide.triggered.connect(self._on_hide_all_curves)

        qt_pos = QPoint(int(event.x),
                        int(self._canvas.height() - event.y))
        menu.exec(self._canvas.mapToGlobal(qt_pos))

    def _on_show_all_curves(self):
        u"""Affiche toutes les courbes."""
        for leg_line, orig_line in self._legend_map.items():
            orig_line.set_visible(True)
            leg_line.set_alpha(1.0)
        self._canvas.draw_idle()

    def _on_hide_all_curves(self):
        u"""Masque toutes les courbes."""
        for leg_line, orig_line in self._legend_map.items():
            orig_line.set_visible(False)
            leg_line.set_alpha(0.2)
        self._canvas.draw_idle()

    def _populate_re_combo(self):
        u"""Peuple le combo Reynolds avec les Re disponibles."""
        self._combo_re.blockSignals(True)
        old_text = self._combo_re.currentText()
        self._combo_re.clear()
        # Collecter les Re depuis les Cp de tous les roles
        re_set = set()
        for sim_results in self._results.values():
            if sim_results.has_cp:
                for k in sim_results.cp:
                    if isinstance(k, float):
                        re_set.add(k)
        for re_val in sorted(re_set):
            self._combo_re.addItem(
                _format_re(re_val), re_val)
        # Restaurer la selection precedente si possible
        idx = self._combo_re.findText(old_text)
        if idx >= 0:
            self._combo_re.setCurrentIndex(idx)
        self._combo_re.blockSignals(False)
        self._combo_re.setVisible(
            self._analysis_name == '-Cp(x)')

    @property
    def _selected_re(self):
        u"""Retourne le Re selectionne ou None."""
        data = self._combo_re.currentData()
        return data

    def _set_combo_value(self, name):
        u"""Positionne le combo sans declencher le signal."""
        self._combo.blockSignals(True)
        idx = self._combo.findText(name)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Zoom molette (centre sur curseur)
    # ------------------------------------------------------------------

    def _on_scroll(self, event):
        u"""Zoom avant/arriere centre sur la position du curseur."""
        if event.inaxes != self._ax:
            return

        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1.0 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        xdata = event.xdata
        ydata = event.ydata
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()

        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        rel_x = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rel_y = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        self._ax.set_xlim(xdata - new_width * rel_x,
                          xdata + new_width * (1 - rel_x))
        self._ax.set_ylim(ydata - new_height * rel_y,
                          ydata + new_height * (1 - rel_y))
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Pan bouton milieu
    # ------------------------------------------------------------------

    def _on_press(self, event):
        u"""Demarre le pan (bouton milieu) ou menu contextuel (bouton droit)."""
        if event.button == 3 and event.inaxes == self._ax:
            self._show_context_menu(event)
            return
        if event.button == 2:
            self._panning = True
            self._pan_start_xy = (event.x, event.y)

    def _on_motion(self, event):
        u"""Deplace la vue pendant le pan, affiche les coordonnees."""
        # Affichage coordonnees curseur
        if event.inaxes == self._ax and event.xdata is not None:
            self._lbl_cursor.setText(
                "x=%.4g  y=%.4g" % (event.xdata, event.ydata))
        else:
            self._lbl_cursor.setText("")

        # Pan
        if not self._panning or self._pan_start_xy is None:
            return

        dx_pix = event.x - self._pan_start_xy[0]
        dy_pix = event.y - self._pan_start_xy[1]
        self._pan_start_xy = (event.x, event.y)

        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        bbox = self._ax.get_window_extent()
        dx_data = -dx_pix * (xlim[1] - xlim[0]) / bbox.width
        dy_data = -dy_pix * (ylim[1] - ylim[0]) / bbox.height

        self._ax.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
        self._ax.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
        self._canvas.draw_idle()

    def _on_release(self, event):
        u"""Arrete le pan."""
        if self._panning:
            self._panning = False
            self._pan_start_xy = None
