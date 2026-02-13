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


# Couleurs
COLOR_CURRENT = '#1f77b4'      # bleu
COLOR_REFERENCE = '#d62728'    # rouge
COLOR_CTRL_POLYGON = '#888888' # gris
COLOR_MEAN_LINE = '#2ca02c'    # vert
COLOR_PORCUPINE = '#ff7f0e'    # orange
PICK_RADIUS = 8                # pixels pour la detection clic


class ProfilCanvas(FigureCanvasQTAgg):
    """Canvas matplotlib pour l'affichage et l'edition interactive de profils."""

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

        # Donnees modele
        self._profil_current = None
        self._profil_reference = None
        self._show_current = True
        self._show_reference = True
        self._show_porc_current = True
        self._show_porc_reference = True

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

        # --- Points de controle (draggables) ---
        self._scatter_ctrl_ext, = self._ax.plot(
            [], [], 's', color=COLOR_CURRENT, markersize=6, picker=PICK_RADIUS)
        self._scatter_ctrl_int, = self._ax.plot(
            [], [], 's', color=COLOR_CURRENT, markersize=6, picker=PICK_RADIUS)

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

        # --- Echelle porcupines ---
        self._porcupine_scale = 1.0

        # --- Drag state ---
        self._dragging = False
        self._drag_bezier = None     # 'ext' ou 'int'
        self._drag_index = None      # index du point de controle
        self._drag_start_xy = None   # position souris au debut du drag

        # --- Connecter les evenements souris ---
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)

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
        self.draw_idle()

    def set_show_porcupines_reference(self, visible):
        """Affiche/masque les porcupines du profil de reference."""
        self._show_porc_reference = visible
        self._porc_ref_ext.set_visible(visible)
        self._porc_ref_int.set_visible(visible)
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

        if p.has_beziers:
            b_ext = p.bezier_extrados
            b_int = p.bezier_intrados

            # Courbes
            pts_ext = b_ext.points
            pts_int = b_int.points
            self._line_current_ext.set_data(pts_ext[:, 0], pts_ext[:, 1])
            self._line_current_int.set_data(pts_int[:, 0], pts_int[:, 1])

            # Points de controle
            cp_ext = b_ext.control_points
            cp_int = b_int.control_points
            self._scatter_ctrl_ext.set_data(cp_ext[:, 0], cp_ext[:, 1])
            self._scatter_ctrl_int.set_data(cp_int[:, 0], cp_int[:, 1])

            # Polygone de controle
            self._line_ctrl_ext.set_data(cp_ext[:, 0], cp_ext[:, 1])
            self._line_ctrl_int.set_data(cp_int[:, 0], cp_int[:, 1])

            # Porcupines
            if self._show_porc_current:
                self._update_porcupines(
                    b_ext, self._porc_current_ext, p.chord,
                    self._env_current_ext)
                self._update_porcupines(
                    b_int, self._porc_current_int, p.chord,
                    self._env_current_int)
            else:
                self._porc_current_ext.set_segments([])
                self._porc_current_int.set_segments([])
                self._env_current_ext.set_data([], [])
                self._env_current_int.set_data([], [])

            # Ligne moyenne
            self._update_mean_line(p, self._line_mean_current)
        else:
            # Mode discret : juste la courbe
            ext = p.extrados
            intr = p.intrados
            self._line_current_ext.set_data(ext[:, 0], ext[:, 1])
            self._line_current_int.set_data(intr[:, 0], intr[:, 1])
            self._clear_ctrl_artists()

    def _update_reference(self):
        """Met a jour tous les artists du profil de reference."""
        p = self._profil_reference
        if p is None or not self._show_reference:
            self._clear_reference_artists()
            return

        if p.has_beziers:
            b_ext = p.bezier_extrados
            b_int = p.bezier_intrados
            pts_ext = b_ext.points
            pts_int = b_int.points
            self._line_ref_ext.set_data(pts_ext[:, 0], pts_ext[:, 1])
            self._line_ref_int.set_data(pts_int[:, 0], pts_int[:, 1])
            if self._show_porc_reference:
                self._update_porcupines(
                    b_ext, self._porc_ref_ext, p.chord,
                    self._env_ref_ext)
                self._update_porcupines(
                    b_int, self._porc_ref_int, p.chord,
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

    def _update_porcupines(self, bezier, collection, chord,
                           envelope_line=None):
        """Calcule et met a jour les porcupines pour une courbe Bezier.

        :param bezier: courbe Bezier
        :param collection: LineCollection a mettre a jour
        :param chord: corde du profil (pour l'echelle)
        :param envelope_line: Line2D pour l'enveloppe (optionnel)
        """
        pts = bezier.points
        normals = bezier.normals
        curvatures = bezier.curvatures

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

    # ==================================================================
    # Drag & drop des points de controle
    # ==================================================================

    def _on_press(self, event):
        """Detecte un clic sur un point de controle ou menu contextuel."""
        if event.button == 3 and event.inaxes == self._ax:
            self._show_context_menu(event)
            return
        if event.button != 1 or event.inaxes != self._ax:
            return
        p = self._profil_current
        if p is None or not p.has_beziers:
            return

        # Chercher le point de controle le plus proche
        idx, side = self._find_nearest_cpoint(event)
        if idx is not None:
            self._dragging = True
            self._drag_bezier = side
            self._drag_index = idx
            self._drag_start_xy = np.array([event.xdata, event.ydata])

    def _on_motion(self, event):
        """Deplace le point de controle pendant le drag."""
        if not self._dragging or event.inaxes != self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        p = self._profil_current
        bez = (p.bezier_extrados if self._drag_bezier == 'ext'
               else p.bezier_intrados)

        current_xy = np.array([event.xdata, event.ydata])
        delta = current_xy - self._drag_start_xy
        self._drag_start_xy = current_xy

        # P1 (a cote du BA) : deplacement vertical uniquement
        if self._drag_index == 1:
            delta[0] = 0.0

        # Deplacer le point de controle
        bez.translate_cpoint(self._drag_index, delta)

        # Mise a jour temps reel
        self._update_current()
        self.draw_idle()

    def _on_release(self, event):
        """Finalise le drag."""
        self._dragging = False
        self._drag_bezier = None
        self._drag_index = None
        self._drag_start_xy = None

    def _find_nearest_cpoint(self, event):
        """Trouve le point de controle le plus proche du clic.

        :returns: (index, 'ext'|'int') ou (None, None)
        """
        p = self._profil_current
        if not p.has_beziers:
            return None, None

        best_dist = PICK_RADIUS  # en pixels
        best_idx = None
        best_side = None

        for side, bez in [('ext', p.bezier_extrados),
                          ('int', p.bezier_intrados)]:
            cp = bez.control_points
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
    # Menu contextuel
    # ==================================================================

    def _show_context_menu(self, event):
        """Affiche le menu contextuel (clic droit)."""
        from PySide6.QtWidgets import QMenu, QInputDialog
        from PySide6.QtCore import QPoint

        menu = QMenu(self)

        # --- Echelle courbure ---
        act_scale = menu.addAction(
            "Echelle courbure (%.2f)..." % self._porcupine_scale)
        act_scale.triggered.connect(self._on_change_porcupine_scale)

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

    # ==================================================================
    # Utilitaires
    # ==================================================================

    def _current_artists(self):
        """Retourne la liste des artists du profil courant."""
        return [
            self._line_current_ext, self._line_current_int,
            self._line_ctrl_ext, self._line_ctrl_int,
            self._scatter_ctrl_ext, self._scatter_ctrl_int,
            self._porc_current_ext, self._porc_current_int,
            self._env_current_ext, self._env_current_int,
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
        self._line_mean_current.set_data([], [])

    def _clear_ctrl_artists(self):
        """Vide les artists de controle (polygone + points)."""
        self._line_ctrl_ext.set_data([], [])
        self._line_ctrl_int.set_data([], [])
        self._scatter_ctrl_ext.set_data([], [])
        self._scatter_ctrl_int.set_data([], [])
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

    def _update_axes(self):
        """Recalcule les limites des axes."""
        self._ax.relim()
        self._ax.autoscale_view()
