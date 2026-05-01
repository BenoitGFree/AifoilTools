#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canvas matplotlib interactif pour l'edition de profils aerodynamiques.

Affiche :
- Profil courant (bleu) et reference (rouge)
- Points de controle Bezier (draggables sur le profil courant)
- Polygone de controle
- Porcupines (courbure le long des normales)
- Ligne moyenne

Le drag des points de controle met a jour tous les elements en temps reel.
"""

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from PySide6.QtCore import Signal


# Couleurs
COLOR_CURRENT = '#1f77b4'      # bleu
COLOR_REFERENCE = '#d62728'    # rouge
COLOR_CTRL_POLYGON = '#888888' # gris
COLOR_MEAN_LINE = '#2ca02c'    # vert
COLOR_PORCUPINE = '#ff7f0e'    # orange
COLOR_DEVIATION = '#000000'     # noir
PICK_RADIUS = 8                # pixels pour la detection clic


class ProfilCanvas(FigureCanvasQTAgg):
    """Canvas matplotlib pour l'affichage et l'edition interactive de profils."""

    profil_edited = Signal()  # emis a la fin d'un drag de point de controle

    def __init__(self, parent=None):
        self._fig = Figure(figsize=(10, 4), dpi=100)
        super().__init__(self._fig)
        self.setParent(parent)

        self._ax = self._fig.add_subplot(111)
        self._ax.set_aspect('equal', adjustable='datalim')
        self._ax.set_xlabel('x (mm)')
        self._ax.set_ylabel('y (mm)')
        self._ax.grid(True, alpha=0.3)
        self._fig.tight_layout()

        self.setToolTip(
            u"Interactions souris :\n"
            u"  - Clic gauche + drag : zoom rectangle (selection d'une"
            u" zone)\n"
            u"  - Clic gauche sur un point de controle : deplacement\n"
            u"  - Molette : zoom centre sur le curseur\n"
            u"  - Clic milieu / Shift+clic gauche : pan (translation)\n"
            u"  - Clic droit : menu contextuel (split, echelle"
            u" porcupines, deviation...)")

        # Donnees modele
        self._profil_current = None
        self._profil_reference = None
        self._show_current = True
        self._show_reference = True
        self._show_porc_current = False
        self._show_porc_reference = False
        self._show_sample_pts = False

        # --- Artists profil courant (bleu) ---
        self._line_current_ext, = self._ax.plot(
            [], [], '-', color=COLOR_CURRENT, linewidth=1.5, label='Courant')
        self._line_current_int, = self._ax.plot(
            [], [], '-', color=COLOR_CURRENT, linewidth=1.5)

        # --- Artists profil reference (rouge) ---
        self._line_ref_ext, = self._ax.plot(
            [], [], '-', color=COLOR_REFERENCE, linewidth=1.2, label=u'R\u00e9f\u00e9rence')
        self._line_ref_int, = self._ax.plot(
            [], [], '-', color=COLOR_REFERENCE, linewidth=1.2)

        # --- Polygone de controle (gris pointille) ---
        self._line_ctrl_ext, = self._ax.plot(
            [], [], '--', color=COLOR_CTRL_POLYGON, linewidth=0.8, alpha=0.6)
        self._line_ctrl_int, = self._ax.plot(
            [], [], '--', color=COLOR_CTRL_POLYGON, linewidth=0.8, alpha=0.6)

        # --- Points de controle interieurs (carres) ---
        self._scatter_ctrl_ext, = self._ax.plot(
            [], [], 's', color=COLOR_CURRENT, markersize=6, picker=PICK_RADIUS)
        self._scatter_ctrl_int, = self._ax.plot(
            [], [], 's', color=COLOR_CURRENT, markersize=6, picker=PICK_RADIUS)

        # --- Points de jonction (extremites de segments : point + cercle) ---
        self._scatter_junc_ext, = self._ax.plot(
            [], [], 'o', color=COLOR_CURRENT, markersize=4)
        self._junc_ring_ext, = self._ax.plot(
            [], [], 'o', markerfacecolor='none',
            markeredgecolor=COLOR_CURRENT, markersize=10, markeredgewidth=1.5)
        self._scatter_junc_int, = self._ax.plot(
            [], [], 'o', color=COLOR_CURRENT, markersize=4)
        self._junc_ring_int, = self._ax.plot(
            [], [], 'o', markerfacecolor='none',
            markeredgecolor=COLOR_CURRENT, markersize=10, markeredgewidth=1.5)

        # --- Porcupines (LineCollection) ---
        self._porc_current_ext = LineCollection(
            [], colors=COLOR_PORCUPINE, linewidths=0.7)
        self._ax.add_collection(self._porc_current_ext)
        self._porc_current_int = LineCollection(
            [], colors=COLOR_PORCUPINE, linewidths=0.7)
        self._ax.add_collection(self._porc_current_int)
        self._porc_ref_ext = LineCollection(
            [], colors=COLOR_REFERENCE, linewidths=0.5, alpha=0.4)
        self._ax.add_collection(self._porc_ref_ext)
        self._porc_ref_int = LineCollection(
            [], colors=COLOR_REFERENCE, linewidths=0.5, alpha=0.4)
        self._ax.add_collection(self._porc_ref_int)

        # --- Enveloppes courbure (ligne reliant les sommets des porcupines) ---
        self._env_current_ext, = self._ax.plot(
            [], [], '-', color=COLOR_CURRENT, linewidth=0.5)
        self._env_current_int, = self._ax.plot(
            [], [], '-', color=COLOR_CURRENT, linewidth=0.5)
        self._env_ref_ext, = self._ax.plot(
            [], [], '-', color=COLOR_REFERENCE, linewidth=0.4, alpha=0.4)
        self._env_ref_int, = self._ax.plot(
            [], [], '-', color=COLOR_REFERENCE, linewidth=0.4, alpha=0.4)

        # --- Ligne moyenne ---
        self._line_mean_current, = self._ax.plot(
            [], [], '--', color=COLOR_CURRENT, linewidth=0.8, alpha=0.7)
        self._line_mean_ref, = self._ax.plot(
            [], [], '--', color=COLOR_REFERENCE, linewidth=0.6, alpha=0.3)

        # --- Points echantillonnes (marqueurs x) ---
        self._sample_pts_ext, = self._ax.plot(
            [], [], 'x', color=COLOR_CURRENT, markersize=4, alpha=0.6)
        self._sample_pts_int, = self._ax.plot(
            [], [], 'x', color=COLOR_CURRENT, markersize=4, alpha=0.6)
        self._sample_pts_ext.set_visible(False)
        self._sample_pts_int.set_visible(False)

        # --- Echelle et densite porcupines ---
        self._porcupine_scale = 200.0
        self._porcupine_n_quills = 200  # nombre de quills par courbe

        # --- Deviation (ecart entre profil courant et reference) ---
        self._show_deviation = False
        self._deviation_scale = 1.0
        self._deviation_n_quills = 200

        self._dev_ext = LineCollection(
            [], colors=COLOR_DEVIATION, linewidths=0.7)
        self._ax.add_collection(self._dev_ext)
        self._dev_int = LineCollection(
            [], colors=COLOR_DEVIATION, linewidths=0.7)
        self._ax.add_collection(self._dev_int)

        self._dev_env_ext, = self._ax.plot(
            [], [], '-', color=COLOR_DEVIATION, linewidth=0.8)
        self._dev_env_int, = self._ax.plot(
            [], [], '-', color=COLOR_DEVIATION, linewidth=0.8)

        # --- Drag state ---
        self._dragging = False
        self._drag_bezier = None     # 'ext' ou 'int'
        self._drag_index = None      # index du point de controle
        self._drag_start_xy = None   # position souris au debut du drag

        # --- Pan state (bouton milieu) ---
        self._panning = False
        self._pan_start_xy = None    # position souris en pixels au debut du pan

        # --- Zoom rectangle (clic gauche) ---
        self._zoom_rect_active = False
        self._zoom_rect_origin = None  # (xdata, ydata) du coin initial
        self._zoom_rect_patch = None   # Rectangle matplotlib

        # --- Connecter les evenements souris ---
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('scroll_event', self._on_scroll)

    # ==================================================================
    # API publique
    # ==================================================================

    def set_current_profil(self, profil):
        """Definit le profil courant (editable).

        :param profil: profil en mode Bezier
        :type profil: model.profil.Profil or None
        """
        self._profil_current = profil
        self._update_current()
        self._update_axes()
        self.draw_idle()

    def set_reference_profil(self, profil):
        """Definit le profil de reference (lecture seule).

        :param profil: profil (Bezier ou discret)
        :type profil: model.profil.Profil or None
        """
        self._profil_reference = profil
        self._update_reference()
        self._update_axes()
        self.draw_idle()

    def set_show_current(self, visible):
        """Affiche/masque le profil courant."""
        self._show_current = visible
        self._set_artists_visible(self._current_artists(), visible)
        self.draw_idle()

    def set_show_reference(self, visible):
        """Affiche/masque le profil de reference."""
        self._show_reference = visible
        self._set_artists_visible(self._reference_artists(), visible)
        self.draw_idle()

    def set_show_porcupines_current(self, visible):
        """Affiche/masque les porcupines du profil courant."""
        self._show_porc_current = visible
        self._porc_current_ext.set_visible(visible)
        self._porc_current_int.set_visible(visible)
        self._update_current()
        self.draw_idle()

    def set_show_porcupines_reference(self, visible):
        """Affiche/masque les porcupines du profil de reference."""
        self._show_porc_reference = visible
        self._porc_ref_ext.set_visible(visible)
        self._porc_ref_int.set_visible(visible)
        self._update_reference()
        self.draw_idle()

    def set_show_sample_points(self, visible):
        """Affiche/masque les points echantillonnes."""
        self._show_sample_pts = visible
        self._update_current()
        self.draw_idle()

    def set_show_deviation(self, visible):
        """Affiche/masque la deviation entre profils."""
        self._show_deviation = visible
        self._update_deviation()
        self.draw_idle()

    def zoom_fit(self):
        """Ajuste le zoom pour voir tout le profil."""
        self._ax.relim()
        self._ax.autoscale_view()
        self.draw_idle()

    # ==================================================================
    # Mise a jour des artists
    # ==================================================================

    def _update_current(self):
        """Met a jour tous les artists du profil courant."""
        p = self._profil_current
        if p is None or not self._show_current:
            self._clear_current_artists()
            return

        if p.has_splines:
            s_ext = p.spline_extrados
            s_int = p.spline_intrados

            # Courbes
            pts_ext = s_ext.points
            pts_int = s_int.points
            self._line_current_ext.set_data(pts_ext[:, 0], pts_ext[:, 1])
            self._line_current_int.set_data(pts_int[:, 0], pts_int[:, 1])

            # Points de controle : separer interieurs et jonctions
            cp_ext = s_ext.control_points
            cp_int = s_int.control_points
            junc_ext = self._junction_indices(s_ext)
            junc_int = self._junction_indices(s_int)
            int_ext = [i for i in range(len(cp_ext)) if i not in junc_ext]
            int_int = [i for i in range(len(cp_int)) if i not in junc_int]
            # Interieurs (carres)
            if int_ext:
                self._scatter_ctrl_ext.set_data(
                    cp_ext[int_ext, 0], cp_ext[int_ext, 1])
            else:
                self._scatter_ctrl_ext.set_data([], [])
            if int_int:
                self._scatter_ctrl_int.set_data(
                    cp_int[int_int, 0], cp_int[int_int, 1])
            else:
                self._scatter_ctrl_int.set_data([], [])
            # Jonctions (point + cercle)
            junc_ext_l = sorted(junc_ext)
            junc_int_l = sorted(junc_int)
            self._scatter_junc_ext.set_data(
                cp_ext[junc_ext_l, 0], cp_ext[junc_ext_l, 1])
            self._junc_ring_ext.set_data(
                cp_ext[junc_ext_l, 0], cp_ext[junc_ext_l, 1])
            self._scatter_junc_int.set_data(
                cp_int[junc_int_l, 0], cp_int[junc_int_l, 1])
            self._junc_ring_int.set_data(
                cp_int[junc_int_l, 0], cp_int[junc_int_l, 1])

            # Polygone de controle
            self._line_ctrl_ext.set_data(cp_ext[:, 0], cp_ext[:, 1])
            self._line_ctrl_int.set_data(cp_int[:, 0], cp_int[:, 1])

            # Porcupines
            if self._show_porc_current:
                self._update_porcupines(
                    s_ext, self._porc_current_ext, p.chord,
                    self._env_current_ext)
                self._update_porcupines(
                    s_int, self._porc_current_int, p.chord,
                    self._env_current_int)
            else:
                self._porc_current_ext.set_segments([])
                self._porc_current_int.set_segments([])
                self._env_current_ext.set_data([], [])
                self._env_current_int.set_data([], [])

            # Points echantillonnes
            if self._show_sample_pts:
                self._sample_pts_ext.set_data(
                    pts_ext[:, 0], pts_ext[:, 1])
                self._sample_pts_int.set_data(
                    pts_int[:, 0], pts_int[:, 1])
                self._sample_pts_ext.set_visible(True)
                self._sample_pts_int.set_visible(True)
            else:
                self._sample_pts_ext.set_data([], [])
                self._sample_pts_int.set_data([], [])
                self._sample_pts_ext.set_visible(False)
                self._sample_pts_int.set_visible(False)

        else:
            # Mode discret : juste la courbe
            ext = p.extrados
            intr = p.intrados
            self._line_current_ext.set_data(ext[:, 0], ext[:, 1])
            self._line_current_int.set_data(intr[:, 0], intr[:, 1])
            self._clear_ctrl_artists()
            self._sample_pts_ext.set_data([], [])
            self._sample_pts_int.set_data([], [])

        # Ligne moyenne (fonctionne en mode discret et spline)
        self._update_mean_line(p, self._line_mean_current)
        self._update_deviation()

    def _update_reference(self):
        """Met a jour tous les artists du profil de reference."""
        p = self._profil_reference
        if p is None or not self._show_reference:
            self._clear_reference_artists()
            return

        if p.has_splines:
            s_ext = p.spline_extrados
            s_int = p.spline_intrados
            pts_ext = s_ext.points
            pts_int = s_int.points
            self._line_ref_ext.set_data(pts_ext[:, 0], pts_ext[:, 1])
            self._line_ref_int.set_data(pts_int[:, 0], pts_int[:, 1])
            if self._show_porc_reference:
                self._update_porcupines(
                    s_ext, self._porc_ref_ext, p.chord,
                    self._env_ref_ext)
                self._update_porcupines(
                    s_int, self._porc_ref_int, p.chord,
                    self._env_ref_int)
            else:
                self._porc_ref_ext.set_segments([])
                self._porc_ref_int.set_segments([])
                self._env_ref_ext.set_data([], [])
                self._env_ref_int.set_data([], [])
        else:
            ext = p.extrados
            intr = p.intrados
            self._line_ref_ext.set_data(ext[:, 0], ext[:, 1])
            self._line_ref_int.set_data(intr[:, 0], intr[:, 1])
            self._porc_ref_ext.set_segments([])
            self._porc_ref_int.set_segments([])
            self._env_ref_ext.set_data([], [])
            self._env_ref_int.set_data([], [])

        self._update_mean_line(p, self._line_mean_ref)
        self._update_deviation()

    def _update_porcupines(self, spline, collection, chord,
                           envelope_line=None):
        """Calcule et met a jour les porcupines pour une spline.

        :param spline: BezierSpline (ou Bezier)
        :param collection: LineCollection a mettre a jour
        :param chord: corde du profil (pour l'echelle)
        :param envelope_line: Line2D pour l'enveloppe (optionnel)
        """
        n = self._porcupine_n_quills
        t_max = getattr(spline, 'n_segments', 1)
        t = np.linspace(0, t_max, n)
        pts = spline.evaluate(t)
        normals = spline.normal(t)
        curvatures = spline.curvature(t)

        # Echelle : la longueur des quills est proportionnelle a la courbure
        # Signe negatif : les quills pointent vers l'interieur du profil
        scale = chord / 10.0 * self._porcupine_scale
        tips = pts - normals * curvatures[:, None] * scale

        # Segments [start, end] pour chaque quill
        segments = np.stack([pts, tips], axis=1)
        collection.set_segments(segments)

        # Enveloppe : ligne reliant les sommets des porcupines
        if envelope_line is not None:
            envelope_line.set_data(tips[:, 0], tips[:, 1])

    def _update_mean_line(self, profil, line_artist):
        """Calcule et trace la ligne moyenne du profil.

        :param profil: profil source
        :param line_artist: Line2D a mettre a jour
        """
        ext = profil.extrados   # BA -> BF, x croissant
        intr = profil.intrados  # BA -> BF, x croissant

        if len(ext) < 2 or len(intr) < 2:
            line_artist.set_data([], [])
            return

        x_min = max(ext[0, 0], intr[0, 0])
        x_max = min(ext[-1, 0], intr[-1, 0])
        x_common = np.linspace(x_min, x_max, 100)

        y_ext = np.interp(x_common, ext[:, 0], ext[:, 1])
        y_int = np.interp(x_common, intr[:, 0], intr[:, 1])
        y_mean = (y_ext + y_int) / 2.0

        line_artist.set_data(x_common, y_mean)

    def _update_deviation(self):
        """Calcule et affiche la deviation entre profils."""
        p_cur = self._profil_current
        p_ref = self._profil_reference

        if (not self._show_deviation
                or p_cur is None or p_ref is None):
            self._clear_deviation_artists()
            return

        from model.profil_spline import ProfilSpline
        n = self._deviation_n_quills
        dev = ProfilSpline.deviation(p_cur, p_ref, n_points=n)
        scale = self._deviation_scale

        # Extrados
        x_ext = dev['x_ext']
        y_cur_ext = dev['y_current_ext']
        dy_ext = dev['y_reference_ext'] - y_cur_ext
        tips_y_ext = y_cur_ext - dy_ext * scale

        bases_ext = np.column_stack([x_ext, y_cur_ext])
        tips_ext = np.column_stack([x_ext, tips_y_ext])
        self._dev_ext.set_segments(
            np.stack([bases_ext, tips_ext], axis=1))
        self._dev_env_ext.set_data(x_ext, tips_y_ext)

        # Intrados
        x_int = dev['x_int']
        y_cur_int = dev['y_current_int']
        dy_int = dev['y_reference_int'] - y_cur_int
        tips_y_int = y_cur_int - dy_int * scale

        bases_int = np.column_stack([x_int, y_cur_int])
        tips_int = np.column_stack([x_int, tips_y_int])
        self._dev_int.set_segments(
            np.stack([bases_int, tips_int], axis=1))
        self._dev_env_int.set_data(x_int, tips_y_int)

    def _clear_deviation_artists(self):
        """Vide les artists de deviation."""
        self._dev_ext.set_segments([])
        self._dev_int.set_segments([])
        self._dev_env_ext.set_data([], [])
        self._dev_env_int.set_data([], [])

    # ==================================================================
    # Drag & drop des points de controle
    # ==================================================================

    def _on_press(self, event):
        """Detecte un clic sur un point de controle, pan ou menu contextuel."""
        if event.button == 3 and event.inaxes == self._ax:
            self._show_context_menu(event)
            return

        # Bouton milieu : demarrer le pan
        if event.button == 2:
            self._panning = True
            self._pan_start_xy = (event.x, event.y)
            return

        if event.button != 1 or event.inaxes != self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        p = self._profil_current
        picked = False
        if p is not None and p.has_splines:
            # Chercher le point de controle le plus proche
            idx, side = self._find_nearest_cpoint(event)
            if idx is not None:
                self._dragging = True
                self._drag_bezier = side
                self._drag_index = idx
                self._drag_start_xy = np.array([event.xdata, event.ydata])
                picked = True

        if not picked:
            # Demarrer le zoom rectangle
            self._zoom_rect_active = True
            self._zoom_rect_origin = (event.xdata, event.ydata)
            self._zoom_rect_patch = Rectangle(
                (event.xdata, event.ydata), 0, 0,
                linewidth=1, edgecolor='#555555',
                facecolor='#cccccc', alpha=0.3, linestyle='--')
            self._ax.add_patch(self._zoom_rect_patch)

    def _on_motion(self, event):
        """Deplace le point de controle pendant le drag, ou pan la vue."""
        # Pan avec bouton milieu
        if self._panning:
            if self._pan_start_xy is None:
                return
            dx_pix = event.x - self._pan_start_xy[0]
            dy_pix = event.y - self._pan_start_xy[1]
            self._pan_start_xy = (event.x, event.y)

            # Convertir pixels -> coordonnees data
            xlim = self._ax.get_xlim()
            ylim = self._ax.get_ylim()
            # Taille de l'axe en pixels
            bbox = self._ax.get_window_extent()
            dx_data = -dx_pix * (xlim[1] - xlim[0]) / bbox.width
            dy_data = -dy_pix * (ylim[1] - ylim[0]) / bbox.height

            self._ax.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
            self._ax.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
            self.draw_idle()
            return

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
                self.draw_idle()
            return

        if not self._dragging or event.inaxes != self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        p = self._profil_current
        spl = (p.spline_extrados if self._drag_bezier == 'ext'
               else p.spline_intrados)

        current_xy = np.array([event.xdata, event.ydata])
        delta = current_xy - self._drag_start_xy
        self._drag_start_xy = current_xy

        # P1 (a cote du BA) : deplacement vertical uniquement
        if self._drag_index == 1:
            delta[0] = 0.0

        # Deplacer le point de controle
        spl.translate_cpoint(self._drag_index, delta)

        # Mise a jour temps reel
        self._update_current()
        self.draw_idle()

    def _on_release(self, event):
        """Finalise le drag, le pan ou le zoom rectangle."""
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
                # Seuil minimum (eviter un zoom sur un clic simple)
                xlim = self._ax.get_xlim()
                ylim = self._ax.get_ylim()
                min_w = (xlim[1] - xlim[0]) * 0.02
                min_h = (ylim[1] - ylim[0]) * 0.02
                if abs(x1 - x0) > min_w and abs(y1 - y0) > min_h:
                    self._ax.set_xlim(min(x0, x1), max(x0, x1))
                    self._ax.set_ylim(min(y0, y1), max(y0, y1))
            self._zoom_rect_origin = None
            self.draw_idle()
            return

        was_dragging = self._dragging
        self._dragging = False
        self._drag_bezier = None
        self._drag_index = None
        self._drag_start_xy = None
        if was_dragging:
            self.profil_edited.emit()

    def _find_nearest_cpoint(self, event):
        """Trouve le point de controle le plus proche du clic.

        :returns: (index, 'ext'|'int') ou (None, None)
        """
        p = self._profil_current
        if not p.has_splines:
            return None, None

        best_dist = PICK_RADIUS  # en pixels
        best_idx = None
        best_side = None

        for side, spl in [('ext', p.spline_extrados),
                          ('int', p.spline_intrados)]:
            cp = spl.control_points
            n_cp = len(cp)
            for i in range(n_cp):
                # Extremites (P0 et P_last) : non deplacables
                if i == 0 or i == n_cp - 1:
                    continue
                # Convertir coordonnees data -> pixels
                x_pix, y_pix = self._ax.transData.transform(cp[i])
                dist = np.hypot(x_pix - event.x, y_pix - event.y)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
                    best_side = side

        return best_idx, best_side

    # ==================================================================
    # Zoom molette
    # ==================================================================

    def _on_scroll(self, event):
        """Zoom avant/arriere centre sur la position du curseur."""
        if event.inaxes != self._ax:
            return

        # Facteur de zoom
        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1.0 / base_scale  # zoom avant
        elif event.button == 'down':
            scale_factor = base_scale         # zoom arriere
        else:
            return

        # Position du curseur en coordonnees data
        xdata = event.xdata
        ydata = event.ydata

        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()

        # Nouvelles limites centrees sur le curseur
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        # Ratio de la position du curseur dans la fenetre
        rel_x = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rel_y = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        self._ax.set_xlim(xdata - new_width * rel_x,
                          xdata + new_width * (1 - rel_x))
        self._ax.set_ylim(ydata - new_height * rel_y,
                          ydata + new_height * (1 - rel_y))
        self.draw_idle()

    # ==================================================================
    # Menu contextuel
    # ==================================================================

    def _show_context_menu(self, event):
        """Affiche le menu contextuel (clic droit)."""
        from PySide6.QtWidgets import QMenu, QInputDialog
        from PySide6.QtCore import QPoint

        menu = QMenu(self)

        # --- Info point ---
        p = self._profil_current
        if (p is not None and p.has_splines
                and event.xdata is not None
                and event.ydata is not None):
            self._info_event_xy = np.array(
                [event.xdata, event.ydata])
            act_info = menu.addAction("Info point...")
            act_info.triggered.connect(self._on_info_point)
            menu.addSeparator()

        # --- Echelle courbure ---
        act_scale = menu.addAction(
            "Echelle courbure (%.2f)..." % self._porcupine_scale)
        act_scale.triggered.connect(self._on_change_porcupine_scale)

        # --- Densite porcupines ---
        density_menu = menu.addMenu(
            u"Densit\u00e9 (%d quills)" % self._porcupine_n_quills)
        act_density_up = density_menu.addAction(u"Densit\u00e9 \u00d72")
        act_density_up.triggered.connect(self._on_density_double)
        act_density_down = density_menu.addAction(u"Densit\u00e9 \u00f72")
        act_density_down.triggered.connect(self._on_density_halve)

        menu.addSeparator()

        # --- Echelle deviation ---
        act_dev_scale = menu.addAction(
            u"\u00c9chelle d\u00e9viation (%.2f)..."
            % self._deviation_scale)
        act_dev_scale.triggered.connect(
            self._on_change_deviation_scale)

        # --- Densite deviation ---
        dev_menu = menu.addMenu(
            u"Densit\u00e9 d\u00e9viation (%d)"
            % self._deviation_n_quills)
        act_dev_up = dev_menu.addAction(u"Densit\u00e9 \u00d72")
        act_dev_up.triggered.connect(
            self._on_deviation_density_double)
        act_dev_down = dev_menu.addAction(u"Densit\u00e9 \u00f72")
        act_dev_down.triggered.connect(
            self._on_deviation_density_halve)

        # --- Split spline ---
        p = self._profil_current
        if p is not None and p.has_splines:
            menu.addSeparator()
            self._split_event_xy = np.array(
                [event.xdata, event.ydata])
            act_split_ext = menu.addAction("Split extrados")
            act_split_ext.triggered.connect(
                self._on_split_extrados)
            act_split_int = menu.addAction("Split intrados")
            act_split_int.triggered.connect(
                self._on_split_intrados)

        # --- Parametres echantillonnage ---
        p = self._profil_current
        if (p is not None and p.has_splines
                and event.xdata is not None
                and event.ydata is not None):
            menu.addSeparator()
            click = np.array([event.xdata, event.ydata])
            s_ext = p.spline_extrados
            s_int = p.spline_intrados
            t_ext, _, d_ext = s_ext.project(click)
            t_int, _, d_int = s_int.project(click)
            if d_ext <= d_int:
                self._param_spline = s_ext
                self._param_side = 'Extrados'
                t_proj = t_ext
            else:
                self._param_spline = s_int
                self._param_side = 'Intrados'
                t_proj = t_int
            spl = self._param_spline
            k, _ = spl._resolve_t(t_proj)
            self._param_seg_idx = k

            # Sous-menu global (spline entiere)
            param_menu = menu.addMenu(
                u"Param\u00e8tres %s" % self._param_side)
            act_npts = param_menu.addAction(
                "Nombre de points (%d)..." % spl.n_points)
            act_npts.triggered.connect(self._on_change_spline_npoints)
            method_menu = param_menu.addMenu(
                u"M\u00e9thode (%s)" % spl.sample_mode)
            act_curvi = method_menu.addAction("Curviligne")
            act_curvi.setCheckable(True)
            act_curvi.setChecked(spl.sample_mode == 'curvilinear')
            act_curvi.triggered.connect(
                lambda: self._on_set_spline_method('curvilinear'))
            act_adapt = method_menu.addAction("Adaptatif")
            act_adapt.setCheckable(True)
            act_adapt.setChecked(spl.sample_mode == 'adaptive')
            act_adapt.triggered.connect(
                lambda: self._on_set_spline_method('adaptive'))

            # Sous-menu segment (per-segment)
            seg_npts = spl.segment_n_points(k)
            seg_mode = spl.segment_sample_mode(k)
            seg_degree = spl._segments[k].degree
            seg_label = "Segment %d/%d" % (k + 1, spl.n_segments)
            seg_menu = menu.addMenu(
                u"Param\u00e8tres %s \u2013 %s"
                % (self._param_side, seg_label))
            act_seg_degree = seg_menu.addAction(
                u"Degr\u00e9 (%d)..." % seg_degree)
            act_seg_degree.triggered.connect(
                self._on_change_segment_degree)
            act_seg_npts = seg_menu.addAction(
                "Nombre de points (%d)..." % seg_npts)
            act_seg_npts.triggered.connect(
                self._on_change_segment_npoints)
            seg_method_menu = seg_menu.addMenu(
                u"M\u00e9thode (%s)" % seg_mode)
            act_seg_curvi = seg_method_menu.addAction("Curviligne")
            act_seg_curvi.setCheckable(True)
            act_seg_curvi.setChecked(seg_mode == 'curvilinear')
            act_seg_curvi.triggered.connect(
                lambda: self._on_set_segment_method('curvilinear'))
            act_seg_adapt = seg_method_menu.addAction("Adaptatif")
            act_seg_adapt.setCheckable(True)
            act_seg_adapt.setChecked(seg_mode == 'adaptive')
            act_seg_adapt.triggered.connect(
                lambda: self._on_set_segment_method('adaptive'))
            seg_menu.addSeparator()
            act_seg_reset = seg_menu.addAction(
                u"R\u00e9initialiser le segment")
            act_seg_reset.triggered.connect(
                self._on_reset_segment_overrides)

        # Convertir position matplotlib -> position widget Qt
        # matplotlib: y=0 en bas ; Qt: y=0 en haut
        qt_pos = QPoint(int(event.x), int(self.height() - event.y))
        menu.exec(self.mapToGlobal(qt_pos))

    def _on_change_porcupine_scale(self):
        """Ouvre un dialogue pour changer l'echelle des porcupines."""
        from PySide6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getDouble(
            self, "Echelle courbure",
            "Facteur d'echelle :",
            self._porcupine_scale,
            0.01, 1e9, 2)
        if ok:
            self._set_porcupine_scale(value)

    def _set_porcupine_scale(self, scale):
        """Change l'echelle des porcupines et rafraichit."""
        self._porcupine_scale = scale
        self._update_current()
        self._update_reference()
        self.draw_idle()

    def _on_density_double(self):
        """Double le nombre de quills."""
        self._porcupine_n_quills *= 2
        self._update_current()
        self._update_reference()
        self.draw_idle()

    def _on_density_halve(self):
        """Divise le nombre de quills par 2."""
        self._porcupine_n_quills = max(5, self._porcupine_n_quills // 2)
        self._update_current()
        self._update_reference()
        self.draw_idle()

    def _on_change_deviation_scale(self):
        """Ouvre un dialogue pour changer l'echelle de deviation."""
        from PySide6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getDouble(
            self, u"\u00c9chelle d\u00e9viation",
            u"Facteur (1.0 = taille r\u00e9elle) :",
            self._deviation_scale,
            0.01, 1e9, 2)
        if ok:
            self._deviation_scale = value
            self._update_deviation()
            self.draw_idle()

    def _on_deviation_density_double(self):
        """Double le nombre de quills de deviation."""
        self._deviation_n_quills *= 2
        self._update_deviation()
        self.draw_idle()

    def _on_deviation_density_halve(self):
        """Divise le nombre de quills de deviation par 2."""
        self._deviation_n_quills = max(
            5, self._deviation_n_quills // 2)
        self._update_deviation()
        self.draw_idle()

    # ==================================================================
    # Info point
    # ==================================================================

    def _on_info_point(self):
        """Affiche les informations du point le plus proche du clic."""
        from PySide6.QtWidgets import QMessageBox

        p = self._profil_current
        if p is None or not p.has_splines:
            return

        click = self._info_event_xy
        s_ext = p.spline_extrados
        s_int = p.spline_intrados

        # Projeter sur les deux cotes
        t_ext, pt_ext, d_ext = s_ext.project(click)
        t_int, pt_int, d_int = s_int.project(click)

        # Choisir le cote le plus proche
        if d_ext <= d_int:
            side_label = 'Extrados'
            t, pt_curve, spl = t_ext, pt_ext, s_ext
        else:
            side_label = 'Intrados'
            t, pt_curve, spl = t_int, pt_int, s_int

        # Segment et parametre local
        k, t_local = spl._resolve_t(t)
        seg = spl._segments[k]
        seg_npts = spl.segment_n_points(k)
        seg_mode = spl.segment_sample_mode(k)

        # Point d'echantillonnage le plus proche
        pts_sample = spl.points
        dists = np.linalg.norm(pts_sample - click, axis=1)
        i_nearest = int(np.argmin(dists))
        pt_sample = pts_sample[i_nearest]

        # Totaux reels (apres overrides et dedup jonctions)
        pts_ext = s_ext.points
        pts_int = s_int.points

        # Detail par segment (alloc effective)
        seg_details = []
        for i in range(spl.n_segments):
            ni = spl.segment_n_points(i)
            mi = spl.segment_sample_mode(i)
            marker = " <<" if i == k else ""
            seg_details.append(
                u"  seg %d : %d pts  %s%s" % (i + 1, ni, mi, marker))

        text = (
            u"Profil : %s\n"
            u"C\u00f4t\u00e9 : %s\n"
            u"\n"
            u"--- Point th\u00e9orique (BezierSpline) ---\n"
            u"  x = %.6f    y = %.6f\n"
            u"  t global = %.4f\n"
            u"\n"
            u"--- Point \u00e9chantillonn\u00e9 le plus proche ---\n"
            u"  x = %.6f    y = %.6f\n"
            u"  indice = %d / %d\n"
            u"\n"
            u"--- Segment B\u00e9zier ---\n"
            u"  segment %d / %d    degr\u00e9 %d\n"
            u"  t local = %.4f\n"
            u"  pts allou\u00e9s = %d    mode = %s\n"
            u"\n"
            u"--- \u00c9chantillonnage %s ---\n"
            u"%s\n"
            u"  total r\u00e9el = %d  (nominal %d, %d jonctions d\u00e9duites)\n"
            u"\n"
            u"--- Totaux r\u00e9els ---\n"
            u"  extrados : %d pts    intrados : %d pts\n"
        ) % (
            p.name, side_label,
            pt_curve[0], pt_curve[1], t,
            pt_sample[0], pt_sample[1],
            i_nearest, len(pts_sample),
            k + 1, spl.n_segments, seg.degree,
            t_local, seg_npts, seg_mode,
            side_label, '\n'.join(seg_details),
            len(pts_sample), spl.n_points,
            spl.n_segments - 1,
            len(pts_ext), len(pts_int),
        )

        QMessageBox.information(self, "Info point", text)

    # ==================================================================
    # Parametres echantillonnage spline
    # ==================================================================

    def _on_change_spline_npoints(self):
        """Ouvre un dialogue pour changer le nombre de points de la spline."""
        from PySide6.QtWidgets import QInputDialog
        spl = self._param_spline
        value, ok = QInputDialog.getInt(
            self,
            u"\u00c9chantillonnage %s" % self._param_side,
            "Nombre de points :",
            spl.n_points, 10, 10000, 10)
        if ok and value != spl.n_points:
            spl.n_points = value
            self._update_current()
            self.draw_idle()

    def _on_set_spline_method(self, mode):
        """Change la methode d'echantillonnage de la spline."""
        spl = self._param_spline
        if mode != spl.sample_mode:
            spl.sample_mode = mode
            self._update_current()
            self.draw_idle()

    # --- Per-segment ---

    def _on_change_segment_degree(self):
        """Ouvre un dialogue pour changer le degre du segment."""
        from PySide6.QtWidgets import QInputDialog
        spl = self._param_spline
        k = self._param_seg_idx
        current = spl._segments[k].degree
        value, ok = QInputDialog.getInt(
            self,
            u"Degr\u00e9 %s \u2013 Segment %d/%d"
            % (self._param_side, k + 1, spl.n_segments),
            u"Degr\u00e9 :",
            current, 1, 30, 1)
        if ok and value != current:
            spl.set_segment_degree(k, value)
            self._update_current()
            self._update_axes()
            self.draw_idle()
            self.profil_edited.emit()

    def _on_change_segment_npoints(self):
        """Ouvre un dialogue pour changer le nombre de points du segment."""
        from PySide6.QtWidgets import QInputDialog
        spl = self._param_spline
        k = self._param_seg_idx
        current = spl.segment_n_points(k)
        value, ok = QInputDialog.getInt(
            self,
            u"\u00c9chantillonnage %s \u2013 Segment %d/%d"
            % (self._param_side, k + 1, spl.n_segments),
            "Nombre de points :",
            current, 2, 10000, 1)
        if ok and value != current:
            spl.set_segment_n_points(k, value)
            self._update_current()
            self.draw_idle()

    def _on_set_segment_method(self, mode):
        """Change la methode d'echantillonnage du segment."""
        spl = self._param_spline
        k = self._param_seg_idx
        if mode != spl.segment_sample_mode(k):
            spl.set_segment_sample_mode(k, mode)
            self._update_current()
            self.draw_idle()

    def _on_reset_segment_overrides(self):
        """Supprime les overrides du segment selectionne."""
        spl = self._param_spline
        k = self._param_seg_idx
        spl.clear_segment_overrides(k)
        self._update_current()
        self.draw_idle()

    # ==================================================================
    # Split spline
    # ==================================================================

    def _on_split_extrados(self):
        """Split l'extrados au point le plus proche du clic."""
        self._do_split('ext')

    def _on_split_intrados(self):
        """Split l'intrados au point le plus proche du clic."""
        self._do_split('int')

    def _do_split(self, side):
        """Execute le split sur le cote specifie.

        :param side: 'ext' ou 'int'
        """
        p = self._profil_current
        if p is None or not p.has_splines:
            return

        spl = (p.spline_extrados if side == 'ext'
               else p.spline_intrados)
        t, pt, dist = spl.project(self._split_event_xy)

        # Rejeter si trop proche d'un bord ou d'une jonction
        eps = 0.01
        N = spl.n_segments
        if t < eps or t > N - eps:
            return
        t_frac = t - int(t)
        if t_frac < eps or t_frac > 1.0 - eps:
            return

        try:
            new_spline = spl.split(t)
        except ValueError:
            return

        if side == 'ext':
            p._spline_extrados = new_spline
        else:
            p._spline_intrados = new_spline

        self._update_current()
        self._update_axes()
        self.draw_idle()
        self.profil_edited.emit()

    # ==================================================================
    # Utilitaires
    # ==================================================================

    @staticmethod
    def _junction_indices(spline):
        """Retourne les indices globaux des points de jonction d'une spline.

        Ce sont les extremites de chaque segment Bezier (indices 0, d1,
        d1+d2, ...) dans le tableau ``control_points``.

        :param spline: BezierSpline
        :returns: set d'indices
        :rtype: set
        """
        indices = {0}
        offset = 0
        for seg in spline._segments:
            offset += seg.degree
            indices.add(offset)
        return indices

    def _current_artists(self):
        """Retourne la liste des artists du profil courant."""
        return [
            self._line_current_ext, self._line_current_int,
            self._line_ctrl_ext, self._line_ctrl_int,
            self._scatter_ctrl_ext, self._scatter_ctrl_int,
            self._scatter_junc_ext, self._junc_ring_ext,
            self._scatter_junc_int, self._junc_ring_int,
            self._porc_current_ext, self._porc_current_int,
            self._env_current_ext, self._env_current_int,
            self._sample_pts_ext, self._sample_pts_int,
            self._line_mean_current,
        ]

    def _reference_artists(self):
        """Retourne la liste des artists du profil de reference."""
        return [
            self._line_ref_ext, self._line_ref_int,
            self._porc_ref_ext, self._porc_ref_int,
            self._env_ref_ext, self._env_ref_int,
            self._line_mean_ref,
        ]

    def _set_artists_visible(self, artists, visible):
        """Montre ou masque une liste d'artists."""
        for a in artists:
            a.set_visible(visible)

    def _clear_current_artists(self):
        """Vide les donnees des artists courants."""
        self._line_current_ext.set_data([], [])
        self._line_current_int.set_data([], [])
        self._clear_ctrl_artists()
        self._porc_current_ext.set_segments([])
        self._porc_current_int.set_segments([])
        self._env_current_ext.set_data([], [])
        self._env_current_int.set_data([], [])
        self._sample_pts_ext.set_data([], [])
        self._sample_pts_int.set_data([], [])
        self._line_mean_current.set_data([], [])
        self._clear_deviation_artists()

    def _clear_ctrl_artists(self):
        """Vide les artists de controle (polygone + points + jonctions)."""
        self._line_ctrl_ext.set_data([], [])
        self._line_ctrl_int.set_data([], [])
        self._scatter_ctrl_ext.set_data([], [])
        self._scatter_ctrl_int.set_data([], [])
        self._scatter_junc_ext.set_data([], [])
        self._junc_ring_ext.set_data([], [])
        self._scatter_junc_int.set_data([], [])
        self._junc_ring_int.set_data([], [])
        self._porc_current_ext.set_segments([])
        self._porc_current_int.set_segments([])
        self._env_current_ext.set_data([], [])
        self._env_current_int.set_data([], [])

    def _clear_reference_artists(self):
        """Vide les donnees des artists de reference."""
        self._line_ref_ext.set_data([], [])
        self._line_ref_int.set_data([], [])
        self._porc_ref_ext.set_segments([])
        self._porc_ref_int.set_segments([])
        self._env_ref_ext.set_data([], [])
        self._env_ref_int.set_data([], [])
        self._line_mean_ref.set_data([], [])
        self._clear_deviation_artists()

    def _update_axes(self):
        """Recalcule les limites des axes."""
        self._ax.relim()
        self._ax.autoscale_view()
