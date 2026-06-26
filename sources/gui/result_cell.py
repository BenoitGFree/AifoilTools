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
from matplotlib.patches import Rectangle
from .i18n import tr as _


# Couleurs (coherentes avec profil_canvas et tab_results)
COLOR_CURRENT = '#1f77b4'      # bleu
COLOR_REFERENCE = '#d62728'    # rouge
COLOR_FLAP = '#2ca02c'         # vert (profil avec volet)

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


def _plot_finesse_cl(ax, sim_results, color, label_base):
    """Finesse (CL/CD) en fonction de CL."""
    re_list = sim_results.re_list
    single = (len(re_list) == 1)
    for i, re_val in enumerate(re_list):
        polar = sim_results.get_polar(re_val)
        if polar is None:
            continue
        ls = '-' if single else RE_LINESTYLES[i % len(RE_LINESTYLES)]
        lbl = label_base if single else '%s %s' % (label_base, _format_re(re_val))
        cl = polar['CL']
        cd = polar['CD']
        finesse = np.where(cd > 1e-8, cl / cd, 0.0)
        ax.plot(cl, finesse, color=color, linestyle=ls, label=lbl)


def _plot_cl_cd(ax, sim_results, color, label_base):
    """CL en fonction de CD (trainee polaire)."""
    _plot_polar_xy(ax, sim_results, color, label_base, 'CD', 'CL')


def _plot_cm_alpha(ax, sim_results, color, label_base):
    """CM en fonction de alpha."""
    _plot_polar_xy(ax, sim_results, color, label_base, 'alpha', 'CM')


def _plot_cm_cl(ax, sim_results, color, label_base):
    """CM en fonction de CL."""
    _plot_polar_xy(ax, sim_results, color, label_base, 'CL', 'CM')


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
        # Cp en derniere colonne ([x, Cp] ou [x, y, Cp])
        ax.plot(cp_data[:, 0], -cp_data[:, -1],
                color=color, linestyle=ls, linewidth=0.8,
                label=lbl)


def _surface_slice(bl):
    u"""Indices de la surface du profil dans des donnees BL (sillage exclu).

    Le DUMP XFoil prolonge la couche limite du profil par le sillage : une
    serie finale de points situes en aval du bord de fuite (x > 1), ou le
    frottement parietal est identiquement nul (Cf = 0). Ces points ne font
    pas partie de la couche limite *du profil* et faussent les traces :
    queue parasite sur « Profil + CL », pic suivi d'une longue tiree sur
    Dstar/Theta(x). On retire la serie finale contigue de tels points.

    Le critere combine (Cf == 0 ET x > 1) est exact pour XFoil et inoffensif
    pour les donnees FlexFoil (surface seule, sans sillage, x <= 1) : rien
    n'est retire dans ce cas.

    :param bl: dict de donnees couche limite (cles 'x', 'Cf', ...)
    :returns: slice des points de surface (a appliquer a chaque tableau)
    :rtype: slice
    """
    x = bl.get('x')
    cf = bl.get('Cf')
    if x is None or cf is None:
        return slice(None)
    i = len(x)
    while i > 0 and cf[i - 1] == 0.0 and x[i - 1] > 1.0:
        i -= 1
    return slice(0, i)


def _bl_at(alpha_dict, alpha_selected):
    u"""Selectionne les donnees BL d'un Re pour une incidence donnee.

    :param alpha_dict: dict {alpha: donnees BL} pour un Reynolds
    :param alpha_selected: incidence demandee (None = premiere dispo)
    :returns: dict de donnees BL ou None
    """
    if not alpha_dict:
        return None
    keys = sorted(alpha_dict.keys())
    if alpha_selected is None:
        return alpha_dict[keys[0]]
    if alpha_selected in alpha_dict:
        return alpha_dict[alpha_selected]
    nearest = min(keys, key=lambda a: abs(a - alpha_selected))
    return alpha_dict[nearest]


def _plot_bl_var(ax, sim_results, color, label_base, x_key, y_key,
                 alpha_selected=None):
    u"""Trace generique d'une variable BL (y_key) en fonction de x_key.

    Compare tous les Reynolds disponibles a l'incidence selectionnee.
    """
    bl_dict = sim_results.bl
    re_vals = sorted(k for k in bl_dict
                     if isinstance(k, float) and bl_dict[k])
    if not re_vals:
        return
    single = (len(re_vals) == 1)
    for i, re_val in enumerate(re_vals):
        bl = _bl_at(bl_dict[re_val], alpha_selected)
        if bl is None or x_key not in bl or y_key not in bl:
            continue
        ls = '-' if single else RE_LINESTYLES[i % len(RE_LINESTYLES)]
        lbl = label_base if single else (
            '%s %s' % (label_base, _format_re(re_val)))
        # Restreindre a la surface du profil (sillage XFoil exclu)
        sl = _surface_slice(bl)
        ax.plot(bl[x_key][sl], bl[y_key][sl],
                color=color, linestyle=ls, label=lbl)


def _plot_ue_s(ax, sim_results, color, label_base, alpha_selected=None):
    u"""Vitesse de bord de couche limite en fonction de s."""
    _plot_bl_var(ax, sim_results, color, label_base, 's', 'Ue_Vinf',
                 alpha_selected)


def _plot_dstar_s(ax, sim_results, color, label_base, alpha_selected=None):
    u"""Epaisseur de deplacement en fonction de s."""
    _plot_bl_var(ax, sim_results, color, label_base, 's', 'Dstar',
                 alpha_selected)


def _plot_theta_s(ax, sim_results, color, label_base, alpha_selected=None):
    u"""Epaisseur de quantite de mouvement en fonction de s."""
    _plot_bl_var(ax, sim_results, color, label_base, 's', 'Theta',
                 alpha_selected)


def _plot_cf_s(ax, sim_results, color, label_base, alpha_selected=None):
    u"""Coefficient de frottement en fonction de s."""
    _plot_bl_var(ax, sim_results, color, label_base, 's', 'Cf',
                 alpha_selected)


def _plot_h_s(ax, sim_results, color, label_base, alpha_selected=None):
    u"""Facteur de forme en fonction de s."""
    _plot_bl_var(ax, sim_results, color, label_base, 's', 'H',
                 alpha_selected)


def _plot_ue_x(ax, sim_results, color, label_base, alpha_selected=None):
    u"""Vitesse de bord de couche limite en fonction de x."""
    _plot_bl_var(ax, sim_results, color, label_base, 'x', 'Ue_Vinf',
                 alpha_selected)


def _plot_dstar_x(ax, sim_results, color, label_base, alpha_selected=None):
    u"""Epaisseur de deplacement en fonction de x."""
    _plot_bl_var(ax, sim_results, color, label_base, 'x', 'Dstar',
                 alpha_selected)


def _plot_theta_x(ax, sim_results, color, label_base, alpha_selected=None):
    u"""Epaisseur de quantite de mouvement en fonction de x."""
    _plot_bl_var(ax, sim_results, color, label_base, 'x', 'Theta',
                 alpha_selected)


def _plot_cf_x(ax, sim_results, color, label_base, alpha_selected=None):
    u"""Coefficient de frottement en fonction de x."""
    _plot_bl_var(ax, sim_results, color, label_base, 'x', 'Cf',
                 alpha_selected)


def _plot_h_x(ax, sim_results, color, label_base, alpha_selected=None):
    u"""Facteur de forme en fonction de x."""
    _plot_bl_var(ax, sim_results, color, label_base, 'x', 'H',
                 alpha_selected)


def _plot_profil_bl(ax, sim_results, color, label_base,
                    re_selected=None, alpha_selected=None):
    u"""Profil tourne par l'incidence + frontiere couche limite.

    Trace le profil (depuis les donnees BL) tourne de alpha degres,
    et la frontiere de couche limite (surface + delta* le long des
    normales sortantes) en jaune.

    :param re_selected: Reynolds selectionne (None = premier dispo)
    :param alpha_selected: incidence en degres (None = 0)
    """
    if not sim_results.has_bl:
        return

    bl_dict = sim_results.bl
    re_vals = sorted(k for k in bl_dict
                     if isinstance(k, float) and bl_dict[k])
    if not re_vals:
        return
    re_val = (re_selected
              if re_selected in bl_dict and bl_dict.get(re_selected)
              else re_vals[0])
    bl = _bl_at(bl_dict[re_val], alpha_selected)
    if bl is None:
        return

    # Restreindre a la surface du profil : le sillage XFoil (x > 1, Cf = 0)
    # n'a pas de normale sortante pertinente et produit une queue parasite
    # ainsi qu'une frontiere delta* deformee en aval du bord de fuite.
    sl = _surface_slice(bl)
    x = bl['x'][sl]
    y = bl['y'][sl]
    dstar = bl['Dstar'][sl]

    # Rotation par l'angle d'incidence
    alpha_deg = alpha_selected if alpha_selected is not None else 0.0
    alpha_rad = np.radians(alpha_deg)
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    xr = x * cos_a + y * sin_a
    yr = -x * sin_a + y * cos_a

    # Normales sortantes par differences finies
    dx = np.gradient(xr)
    dy = np.gradient(yr)
    norm = np.sqrt(dx**2 + dy**2)
    norm = np.where(norm < 1e-15, 1.0, norm)
    # Convention Selig (BF->ext->BA->int->BF) : (dy, -dx) pointe
    # vers l'exterieur du profil
    nx = dy / norm
    ny = -dx / norm

    # Frontiere couche limite
    xbl = xr + nx * dstar
    ybl = yr + ny * dstar

    # Tracer le profil
    ax.plot(xr, yr, color=color, linewidth=1.2, label=label_base)

    # Frontiere CL en jaune
    ax.plot(xbl, ybl, color='#FFD700', linewidth=1.0,
            label=u'%s \u03b4*' % label_base)

    ax.set_aspect('equal', adjustable='datalim')


def _cp_select(sim_results, re_selected, alpha_selected):
    u"""Resout (re_val, alpha, cp_data) pour une distribution Cp.

    :returns: tuple (re_val, alpha, cp_data) ou None si indisponible
    """
    if not sim_results.has_cp:
        return None
    cp_dict = sim_results.cp
    re_vals = sorted(k for k in cp_dict if isinstance(k, float))
    if not re_vals:
        return None
    re_val = re_selected if re_selected in cp_dict else re_vals[0]
    alphas = sorted(cp_dict[re_val].keys())
    if not alphas:
        return None
    if alpha_selected is None:
        alpha = alphas[0]
    else:
        # alpha disponible le plus proche de la selection
        alpha = min(alphas, key=lambda a: abs(a - alpha_selected))
    cp_data = cp_dict[re_val].get(alpha)
    if cp_data is None or len(cp_data) == 0:
        return None
    return re_val, alpha, cp_data


def _coeffs_at(sim_results, re_val, alpha):
    u"""Retourne (CL, CM, CD, finesse) au point (re, alpha) le plus proche.

    Les valeurs absentes sont renvoyees a None.

    :rtype: dict
    """
    out = {'CL': None, 'CM': None, 'CD': None, 'LD': None}
    polar = sim_results.get_polar(re_val)
    if polar is None or 'alpha' not in polar or len(polar['alpha']) == 0:
        return out
    a_arr = np.asarray(polar['alpha'], dtype=float)
    idx = int(np.argmin(np.abs(a_arr - alpha)))
    cl = float(polar['CL'][idx]) if 'CL' in polar else None
    cd = float(polar['CD'][idx]) if 'CD' in polar else None
    cm = float(polar['CM'][idx]) if 'CM' in polar else None
    out['CL'] = cl
    out['CM'] = cm
    out['CD'] = cd
    if cl is not None and cd is not None and cd > 1e-8:
        out['LD'] = cl / cd
    return out


# Position verticale de l'encart de coefficients selon le role (couleur),
# pour empiler les profils compares sans chevauchement.
_CP_BOX_SLOT = {COLOR_CURRENT: 0, COLOR_REFERENCE: 1, COLOR_FLAP: 2}


def _plot_cp_xfoil(ax, sim_results, color, label_base,
                   re_selected=None, alpha_selected=None):
    u"""Distribution -Cp(x) facon XFoil : pression en haut, profil en bas.

    Reproduit le trace combine classique de XFoil pour un (Re, alpha)
    donne : la courbe de pression (-Cp, succion vers le haut) et le
    profil redessine en dessous, a l'echelle de corde. Un encart resume
    les coefficients au point choisi (CL, CM, CD, L/D).

    :param re_selected: Reynolds selectionne (None = premier dispo)
    :param alpha_selected: incidence en degres (None = premiere dispo)
    """
    sel = _cp_select(sim_results, re_selected, alpha_selected)
    if sel is None:
        return
    re_val, alpha, cp_data = sel

    x = cp_data[:, 0]
    neg_cp = -cp_data[:, -1]

    # Courbe de pression
    ax.plot(x, neg_cp, color=color, linewidth=1.0,
            label=u'%s α=%.1f°' % (label_base, alpha))

    # Profil redessine en bas, a l'echelle de corde (1 unite de corde =
    # 1 unite de Cp, comme dans la fenetre XFoil). La geometrie (y) n'est
    # presente que si le fichier Cp a 3 colonnes ; sinon on retombe sur
    # la couche limite si disponible.
    xf = yf = None
    if cp_data.shape[1] >= 3:
        xf = x
        yf = cp_data[:, 1]
    elif sim_results.has_bl:
        bl = _bl_at(sim_results.bl.get(re_val), alpha)
        if bl is not None and 'x' in bl and 'y' in bl:
            xf = bl['x']
            yf = bl['y']
    if xf is not None and yf is not None and len(yf) > 0:
        margin = 0.20
        # Place le sommet du profil 'margin' sous le bas de la courbe.
        y_base = float(np.min(neg_cp)) - margin - float(np.max(yf))
        ax.plot(xf, yf + y_base, color=color, linewidth=1.0)

    # Encart des coefficients
    coeffs = _coeffs_at(sim_results, re_val, alpha)
    slot = _CP_BOX_SLOT.get(color, 0)
    _annotate_coeffs(ax, color, label_base, re_val, alpha, coeffs, slot)


def _annotate_coeffs(ax, color, label_base, re_val, alpha, coeffs, slot):
    u"""Affiche l'encart de coefficients facon XFoil (coin haut-droit)."""
    def _fmt(v, fmt):
        return (fmt % v) if v is not None else u'—'
    lines = [
        label_base,
        u'Re  = %s' % _format_re(re_val).replace('Re=', ''),
        u'α   = %.2f°' % alpha,
        u'CL  = %s' % _fmt(coeffs['CL'], '%.4f'),
        u'CM  = %s' % _fmt(coeffs['CM'], '%.4f'),
        u'CD  = %s' % _fmt(coeffs['CD'], '%.5f'),
        u'L/D = %s' % _fmt(coeffs['LD'], '%.2f'),
    ]
    y0 = 0.98 - slot * 0.30
    ax.text(0.985, y0, '\n'.join(lines),
            transform=ax.transAxes, ha='right', va='top',
            fontsize=6, family='monospace', color=color,
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor=color, alpha=0.75, linewidth=0.6))


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
    ('Finesse(CL)',    _plot_finesse_cl,    'CL',          'CL/CD'),
    ('CL(CD)',         _plot_cl_cd,         'CD',          'CL'),
    ('-Cp(x)',         _plot_cp_x,          'x/c',         '-Cp'),
    ('Cp + Profil',    _plot_cp_xfoil,      'x/c',         '-Cp'),
    ('CM(alpha)',      _plot_cm_alpha,      'alpha (deg)', 'CM'),
    ('CM(CL)',         _plot_cm_cl,         'CL',          'CM'),
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
    ('Profil + CL',    _plot_profil_bl,     'x',           'y'),
]

# Noms pour acces rapide
ANALYSIS_NAMES = [a[0] for a in ANALYSIS_REGISTRY]

# Analyses de couche limite : tracees pour une incidence donnee, en
# comparant tous les Reynolds disponibles.
_BL_ANALYSES = ('Ue(s)', 'Dstar(s)', 'Theta(s)', 'Cf(s)', 'H(s)',
                'Ue(x)', 'Dstar(x)', 'Theta(x)', 'Cf(x)', 'H(x)')

# Analyses dependant d'un seul Reynolds (combo Re visible) et/ou d'une
# seule incidence (combo alpha visible). Les analyses de couche limite
# dependent de l'incidence (depuis le calcul d'un DUMP par alpha) mais
# pas d'un Reynolds unique (elles les comparent tous).
_NEEDS_RE = ('-Cp(x)', 'Cp + Profil', 'Profil + CL')
_NEEDS_ALPHA = ('Cp + Profil', 'Profil + CL') + _BL_ANALYSES


def _analysis_label(name):
    u"""Libelle d'affichage (traduit) d'une analyse.

    L'identifiant interne (nom non traduit) reste la cle utilisee
    partout (registre, lookup, sauvegarde) ; seul l'affichage est
    traduit. Pour la plupart des noms (notation aero CL/CD/alpha...)
    la traduction est l'identite ; seul 'Finesse(...)' change.
    """
    return _(name)

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
        self._show_flap = True
        self._analysis_name = analysis_name

        # Pan state
        self._panning = False
        self._pan_start_xy = None

        # Zoom rectangle state
        self._zoom_rect_active = False
        self._zoom_rect_origin = None
        self._zoom_rect_patch = None

        # Mapping legende -> ligne pour toggle visibilite
        self._legend_map = {}  # {legend_line: original_line}
        self._hidden_labels = set()  # labels des courbes masquees

        # Defilement legende (quand beaucoup d'entrees)
        self._legend_offset = 0
        self._legend_max_visible = 15
        self._all_handles = []
        self._all_labels = []

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
        # Libelle affiche (traduit) en texte, identifiant en userData.
        self._combo.blockSignals(True)
        for nm in ANALYSIS_NAMES:
            self._combo.addItem(_analysis_label(nm), nm)
        self._combo.blockSignals(False)
        self._combo.setToolTip(
            _(u"Type d'analyse affiche dans cette cellule.\n"
            u"Polaires : CL(alpha), CD(alpha), CL/CD, Cm(alpha)...\n"
            u"Distributions : -Cp(x), Cp + Profil, Profil + CL\n"
            u"Couche limite : Ue, Dstar, Theta, Cf, H (vs s ou x)"))
        self._combo.currentIndexChanged.connect(self._on_analysis_changed)
        ctrl.addWidget(self._combo)

        self._chk_current = QCheckBox(_("C"))
        self._chk_current.setChecked(True)
        self._chk_current.setToolTip(
            _(u"Afficher les courbes du profil courant (bleu)"))
        self._chk_current.setStyleSheet("color: %s;" % COLOR_CURRENT)
        self._chk_current.stateChanged.connect(self._on_toggle_current)
        ctrl.addWidget(self._chk_current)

        self._chk_reference = QCheckBox(_("R"))
        self._chk_reference.setChecked(True)
        self._chk_reference.setToolTip(
            _(u"Afficher les courbes du profil de reference (rouge)"))
        self._chk_reference.setStyleSheet(
            "color: %s;" % COLOR_REFERENCE)
        self._chk_reference.stateChanged.connect(
            self._on_toggle_reference)
        ctrl.addWidget(self._chk_reference)

        self._chk_flap = QCheckBox(_("F"))
        self._chk_flap.setChecked(True)
        self._chk_flap.setToolTip(
            _(u"Afficher les courbes du profil avec volet (vert)"))
        self._chk_flap.setStyleSheet("color: %s;" % COLOR_FLAP)
        self._chk_flap.stateChanged.connect(self._on_toggle_flap)
        ctrl.addWidget(self._chk_flap)

        # Combo Reynolds (visible pour -Cp(x) et Profil + CL)
        self._combo_re = QComboBox()
        self._combo_re.setToolTip(
            _(u"Choix du Reynolds pour les analyses dependant d'un seul"
            u" Reynolds (-Cp(x), Profil+CL, couche limite)."))
        self._combo_re.setMinimumWidth(90)
        self._combo_re.currentTextChanged.connect(
            self._on_re_changed)
        self._combo_re.setVisible(False)
        ctrl.addWidget(self._combo_re)

        # Combo Alpha (visible pour Profil + CL)
        self._combo_alpha = QComboBox()
        self._combo_alpha.setToolTip(
            _(u"Choix de l'angle d'incidence (alpha) pour les analyses"
            u" point-par-point (-Cp, Profil+CL, couche limite)."))
        self._combo_alpha.setMinimumWidth(70)
        self._combo_alpha.currentTextChanged.connect(
            self._on_alpha_changed)
        self._combo_alpha.setVisible(False)
        ctrl.addWidget(self._combo_alpha)

        self._lbl_cursor = QLabel("")
        self._lbl_cursor.setStyleSheet("font-size: 9px; color: #666;")
        self._lbl_cursor.setToolTip(
            _(u"Coordonnees du curseur souris dans le repere du graphique."))
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
        self._populate_alpha_combo()
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
        self._ax.clear()
        self._ax.grid(True, alpha=0.3)

        plot_fn, xlabel, ylabel = _get_analysis(self._analysis_name)
        if plot_fn is None:
            self._canvas.draw_idle()
            return

        is_cp = (self._analysis_name == '-Cp(x)')
        is_cp_xfoil = (self._analysis_name == 'Cp + Profil')
        is_profil_bl = (self._analysis_name == 'Profil + CL')
        is_bl = self._analysis_name in _BL_ANALYSES
        for role, sim_results in self._results.items():
            if (is_cp or is_cp_xfoil) and not sim_results.has_cp:
                continue
            if (is_bl or is_profil_bl) \
                    and not sim_results.has_bl:
                continue
            if not is_cp and not is_cp_xfoil and not is_bl \
                    and not is_profil_bl \
                    and not sim_results.has_polars:
                continue
            if role == 'current' and not self._show_current:
                continue
            if role == 'reference' \
                    and not self._show_reference:
                continue
            if role == 'flap' and not self._show_flap:
                continue
            color = {'current': COLOR_CURRENT,
                     'reference': COLOR_REFERENCE,
                     'flap': COLOR_FLAP}.get(role, COLOR_CURRENT)
            label_base = {'current': 'Courant',
                          'reference': u'R\u00e9f.',
                          'flap': 'Flap'}.get(role, role)
            if is_cp:
                plot_fn(self._ax, sim_results, color,
                        label_base, self._selected_re)
            elif is_profil_bl or is_cp_xfoil:
                plot_fn(self._ax, sim_results, color,
                        label_base, self._selected_re,
                        self._selected_alpha)
            elif is_bl:
                plot_fn(self._ax, sim_results, color,
                        label_base, self._selected_alpha)
            else:
                plot_fn(self._ax, sim_results, color,
                        label_base)

        self._ax.set_xlabel(xlabel, fontsize=8)
        self._ax.set_ylabel(ylabel, fontsize=8)
        title = _analysis_label(self._analysis_name)
        if is_cp and self._selected_re is not None:
            title += '  %s' % _format_re(self._selected_re)
        if is_profil_bl or is_cp_xfoil:
            if self._selected_re is not None:
                title += '  %s' % _format_re(
                    self._selected_re)
            if self._selected_alpha is not None:
                title += u'  \u03b1=%.1f\u00b0' % (
                    self._selected_alpha)
        if is_bl and self._selected_alpha is not None:
            title += u'  \u03b1=%.1f\u00b0' % self._selected_alpha
        self._ax.set_title(title, fontsize=9)
        self._ax.tick_params(labelsize=7)

        # Legende interactive avec defilement
        self._all_handles, self._all_labels = \
            self._ax.get_legend_handles_labels()
        if self._all_labels:
            n = len(self._all_labels)
            if n <= self._legend_max_visible:
                self._legend_offset = 0
            else:
                self._legend_offset = max(
                    0, min(self._legend_offset,
                           n - self._legend_max_visible))
            # Appliquer l'etat masque aux lignes
            for handle in self._all_handles:
                if handle.get_label() in self._hidden_labels:
                    handle.set_visible(False)
            self._build_legend()
        else:
            self._legend_map = {}

        self._canvas.draw_idle()

    def _build_legend(self):
        u"""Construit la legende avec fenetre de defilement."""
        leg = self._ax.get_legend()
        if leg is not None:
            leg.remove()

        n = len(self._all_labels)
        if n == 0:
            self._legend_map = {}
            return

        max_vis = self._legend_max_visible
        offset = self._legend_offset

        if n <= max_vis:
            vis_handles = self._all_handles
            vis_labels = self._all_labels
            title = None
        else:
            end = min(offset + max_vis, n)
            vis_handles = self._all_handles[offset:end]
            vis_labels = self._all_labels[offset:end]
            parts = []
            if offset > 0:
                parts.append(u'\u25b2 %d' % offset)
            if end < n:
                parts.append(u'\u25bc %d' % (n - end))
            title = '   '.join(parts) if parts else None

        legend = self._ax.legend(
            vis_handles, vis_labels, fontsize=6,
            title=title, title_fontsize=7)

        self._legend_map = {}
        for leg_line, orig_handle in zip(
                legend.get_lines(), vis_handles):
            leg_line.set_visible(True)  # toujours cliquable
            leg_line.set_picker(5)
            self._legend_map[leg_line] = orig_handle
            if orig_handle.get_label() \
                    in self._hidden_labels:
                leg_line.set_alpha(0.2)

    # ------------------------------------------------------------------
    # Slots controles
    # ------------------------------------------------------------------

    def _on_analysis_changed(self, _index=None):
        u"""Appele quand le combo d'analyse change."""
        name = self._combo.currentData()
        if name is None:
            return
        self._analysis_name = name
        self._legend_offset = 0
        self._combo_re.setVisible(name in _NEEDS_RE)
        needs_alpha = name in _NEEDS_ALPHA
        self._combo_alpha.setVisible(needs_alpha)
        if needs_alpha:
            self._populate_alpha_combo()
        self._replot()

    def _on_re_changed(self, text):
        u"""Appele quand le combo Reynolds change."""
        self._replot()

    def _on_alpha_changed(self, text):
        u"""Appele quand le combo alpha change."""
        self._replot()

    def _on_toggle_current(self, state):
        self._show_current = (state == Qt.Checked.value)
        self._replot()

    def _on_toggle_reference(self, state):
        self._show_reference = (state == Qt.Checked.value)
        self._replot()

    def _on_toggle_flap(self, state):
        self._show_flap = (state == Qt.Checked.value)
        self._replot()

    def _on_pick_legend(self, event):
        u"""Toggle la visibilite d'une courbe en cliquant sur la legende."""
        leg_line = event.artist
        if leg_line not in self._legend_map:
            return
        orig_line = self._legend_map[leg_line]
        visible = not orig_line.get_visible()
        orig_line.set_visible(visible)
        leg_line.set_alpha(1.0 if visible else 0.2)
        label = orig_line.get_label()
        if visible:
            self._hidden_labels.discard(label)
        else:
            self._hidden_labels.add(label)
        self._canvas.draw_idle()

    def _show_context_menu(self, event):
        u"""Menu contextuel : tout afficher / tout masquer."""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtCore import QPoint

        if not self._legend_map:
            return

        menu = QMenu(self._canvas)
        act_show = menu.addAction(_("Tout afficher"))
        act_hide = menu.addAction(_("Tout masquer"))

        act_show.triggered.connect(self._on_show_all_curves)
        act_hide.triggered.connect(self._on_hide_all_curves)

        qt_pos = QPoint(int(event.x),
                        int(self._canvas.height() - event.y))
        menu.exec(self._canvas.mapToGlobal(qt_pos))

    def _on_show_all_curves(self):
        u"""Affiche toutes les courbes."""
        self._hidden_labels.clear()
        for handle in self._all_handles:
            handle.set_visible(True)
        for leg_line in self._legend_map:
            leg_line.set_alpha(1.0)
        self._canvas.draw_idle()

    def _on_hide_all_curves(self):
        u"""Masque toutes les courbes."""
        self._hidden_labels = set(self._all_labels)
        for handle in self._all_handles:
            handle.set_visible(False)
        for leg_line in self._legend_map:
            leg_line.set_alpha(0.2)
        self._canvas.draw_idle()

    def _populate_re_combo(self):
        u"""Peuple le combo Reynolds avec les Re disponibles."""
        self._combo_re.blockSignals(True)
        old_text = self._combo_re.currentText()
        self._combo_re.clear()
        re_set = set()
        for sim_results in self._results.values():
            if sim_results.has_cp:
                for k in sim_results.cp:
                    if isinstance(k, float):
                        re_set.add(k)
            if sim_results.has_bl:
                for k in sim_results.bl:
                    if isinstance(k, float):
                        re_set.add(k)
        for re_val in sorted(re_set):
            self._combo_re.addItem(
                _format_re(re_val), re_val)
        idx = self._combo_re.findText(old_text)
        if idx >= 0:
            self._combo_re.setCurrentIndex(idx)
        self._combo_re.blockSignals(False)
        self._combo_re.setVisible(self._analysis_name in _NEEDS_RE)

    def _populate_alpha_combo(self):
        u"""Peuple le combo alpha depuis les polaires et les Cp."""
        self._combo_alpha.blockSignals(True)
        old_text = self._combo_alpha.currentText()
        self._combo_alpha.clear()
        alpha_set = set()
        for sim_results in self._results.values():
            if sim_results.has_polars:
                for re_val in sim_results.re_list:
                    polar = sim_results.get_polar(re_val)
                    if polar is not None:
                        for a in polar['alpha']:
                            alpha_set.add(float(a))
            # Les distributions Cp peuvent exister sans polaire
            if sim_results.has_cp:
                for re_val, alphas in sim_results.cp.items():
                    if isinstance(re_val, float):
                        for a in alphas:
                            alpha_set.add(float(a))
            # Idem pour les couches limites (un DUMP par alpha)
            if sim_results.has_bl:
                for re_val, alphas in sim_results.bl.items():
                    if isinstance(re_val, float):
                        for a in alphas:
                            alpha_set.add(float(a))
        for a in sorted(alpha_set):
            self._combo_alpha.addItem(
                _(u'%.1f\u00b0') % a, a)
        idx = self._combo_alpha.findText(old_text)
        if idx >= 0:
            self._combo_alpha.setCurrentIndex(idx)
        self._combo_alpha.blockSignals(False)
        self._combo_alpha.setVisible(
            self._analysis_name in _NEEDS_ALPHA)

    @property
    def _selected_re(self):
        u"""Retourne le Re selectionne ou None."""
        return self._combo_re.currentData()

    @property
    def _selected_alpha(self):
        u"""Retourne l'alpha selectionne ou None."""
        return self._combo_alpha.currentData()

    def _set_combo_value(self, name):
        u"""Positionne le combo sans declencher le signal."""
        self._combo.blockSignals(True)
        idx = self._combo.findData(name)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Zoom molette (centre sur curseur)
    # ------------------------------------------------------------------

    def _on_scroll(self, event):
        u"""Zoom ou defilement legende."""
        # Defilement legende si survol et necessaire
        if len(self._all_labels) > self._legend_max_visible:
            legend = self._ax.get_legend()
            if legend is not None:
                try:
                    renderer = self._canvas.get_renderer()
                    bb = legend.get_window_extent(renderer)
                    if bb.contains(event.x, event.y):
                        n = len(self._all_labels)
                        mx = n - self._legend_max_visible
                        if event.button == 'up':
                            self._legend_offset = max(
                                0, self._legend_offset - 3)
                        elif event.button == 'down':
                            self._legend_offset = min(
                                mx, self._legend_offset + 3)
                        self._build_legend()
                        self._canvas.draw_idle()
                        return
                except Exception:
                    pass

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
        u"""Demarre le pan, zoom rectangle ou menu contextuel."""
        if event.button == 3 and event.inaxes == self._ax:
            self._show_context_menu(event)
            return
        if event.button == 2:
            self._panning = True
            self._pan_start_xy = (event.x, event.y)
            return
        if (event.button == 1 and event.inaxes == self._ax
                and event.xdata is not None
                and event.ydata is not None):
            self._zoom_rect_active = True
            self._zoom_rect_origin = (event.xdata, event.ydata)
            self._zoom_rect_patch = Rectangle(
                (event.xdata, event.ydata), 0, 0,
                linewidth=1, edgecolor='#555555',
                facecolor='#cccccc', alpha=0.3, linestyle='--')
            self._ax.add_patch(self._zoom_rect_patch)

    def _on_motion(self, event):
        u"""Deplace la vue pendant le pan, affiche les coordonnees."""
        # Affichage coordonnees curseur
        if event.inaxes == self._ax and event.xdata is not None:
            self._lbl_cursor.setText(
                _("x=%.4g  y=%.4g") % (event.xdata, event.ydata))
        else:
            self._lbl_cursor.setText("")

        # Zoom rectangle
        if self._zoom_rect_active:
            if event.xdata is not None and event.ydata is not None:
                x0, y0 = self._zoom_rect_origin
                self._zoom_rect_patch.set_xy(
                    (min(x0, event.xdata), min(y0, event.ydata)))
                self._zoom_rect_patch.set_width(
                    abs(event.xdata - x0))
                self._zoom_rect_patch.set_height(
                    abs(event.ydata - y0))
                self._canvas.draw_idle()
            return

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
        u"""Arrete le pan ou applique le zoom rectangle."""
        if self._panning:
            self._panning = False
            self._pan_start_xy = None
            return

        if self._zoom_rect_active:
            self._zoom_rect_active = False
            if self._zoom_rect_patch is not None:
                self._zoom_rect_patch.remove()
                self._zoom_rect_patch = None
            if (self._zoom_rect_origin is not None
                    and event.xdata is not None
                    and event.ydata is not None):
                x0, y0 = self._zoom_rect_origin
                x1, y1 = event.xdata, event.ydata
                xlim = self._ax.get_xlim()
                ylim = self._ax.get_ylim()
                min_w = (xlim[1] - xlim[0]) * 0.02
                min_h = (ylim[1] - ylim[0]) * 0.02
                if abs(x1 - x0) > min_w and abs(y1 - y0) > min_h:
                    self._ax.set_xlim(min(x0, x1), max(x0, x1))
                    self._ax.set_ylim(min(y0, y1), max(y0, y1))
            self._zoom_rect_origin = None
            self._canvas.draw_idle()
