#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Onglet Profils : affichage et edition interactive des profils."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel
)
from PySide6.QtCore import Qt, Signal

from model.profil_spline import ProfilSpline
from .profil_canvas import ProfilCanvas


class TabProfils(QWidget):
    """Onglet principal d'edition des profils aerodynamiques."""

    profil_changed = Signal(str)  # role: 'current' ou 'reference'

    def __init__(self, parent=None):
        super().__init__(parent)

        # Profils
        self._profil_current = None
        self._profil_reference = None

        self._build_ui()
        self._load_default_profiles()

    def _build_ui(self):
        """Construit l'interface de l'onglet."""
        layout = QVBoxLayout(self)

        # --- Barre de controle superieure ---
        ctrl_layout = QHBoxLayout()

        # Checkbox profil courant
        self._chk_current = QCheckBox("Profil courant")
        self._chk_current.setChecked(True)
        self._chk_current.stateChanged.connect(self._on_toggle_current)
        ctrl_layout.addWidget(self._chk_current)

        self._lbl_current = QLabel("NACA 2412")
        self._lbl_current.setStyleSheet("color: #1f77b4; font-weight: bold;")
        ctrl_layout.addWidget(self._lbl_current)

        self._chk_porc_current = QCheckBox("Courbure")
        self._chk_porc_current.setChecked(True)
        self._chk_porc_current.stateChanged.connect(self._on_toggle_porc_current)
        ctrl_layout.addWidget(self._chk_porc_current)

        self._chk_sample_pts = QCheckBox("Pts")
        self._chk_sample_pts.setChecked(False)
        self._chk_sample_pts.stateChanged.connect(
            self._on_toggle_sample_pts)
        ctrl_layout.addWidget(self._chk_sample_pts)

        ctrl_layout.addSpacing(40)

        # Checkbox profil reference
        self._chk_reference = QCheckBox(u"Profil r\u00e9f\u00e9rence")
        self._chk_reference.setChecked(True)
        self._chk_reference.stateChanged.connect(self._on_toggle_reference)
        ctrl_layout.addWidget(self._chk_reference)

        self._lbl_reference = QLabel("NACA 0012")
        self._lbl_reference.setStyleSheet("color: #d62728; font-weight: bold;")
        ctrl_layout.addWidget(self._lbl_reference)

        self._chk_porc_reference = QCheckBox("Courbure")
        self._chk_porc_reference.setChecked(True)
        self._chk_porc_reference.stateChanged.connect(
            self._on_toggle_porc_reference)
        ctrl_layout.addWidget(self._chk_porc_reference)

        ctrl_layout.addSpacing(20)

        self._chk_deviation = QCheckBox(u"D\u00e9viation")
        self._chk_deviation.setChecked(False)
        self._chk_deviation.stateChanged.connect(
            self._on_toggle_deviation)
        ctrl_layout.addWidget(self._chk_deviation)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # --- Canvas matplotlib ---
        self._canvas = ProfilCanvas(self)
        self._canvas.profil_edited.connect(
            lambda: self.profil_changed.emit('current'))
        layout.addWidget(self._canvas, stretch=1)

    def _load_default_profiles(self):
        """Charge les profils par defaut au demarrage (mode discret)."""
        self._profil_current = ProfilSpline.from_naca('2412', n_points=150)
        self._profil_current.normalize()

        self._profil_reference = ProfilSpline.from_naca('0012', n_points=150)
        self._profil_reference.normalize()

        self._canvas.set_current_profil(self._profil_current)
        self._canvas.set_reference_profil(self._profil_reference)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_toggle_current(self, state):
        """Affiche/masque le profil courant."""
        self._canvas.set_show_current(state == Qt.Checked.value)

    def _on_toggle_reference(self, state):
        """Affiche/masque le profil de reference."""
        self._canvas.set_show_reference(state == Qt.Checked.value)

    def _on_toggle_porc_current(self, state):
        """Affiche/masque les porcupines du profil courant."""
        self._canvas.set_show_porcupines_current(state == Qt.Checked.value)

    def _on_toggle_sample_pts(self, state):
        """Affiche/masque les points echantillonnes."""
        self._canvas.set_show_sample_points(
            state == Qt.Checked.value)

    def _on_toggle_porc_reference(self, state):
        """Affiche/masque les porcupines du profil de reference."""
        self._canvas.set_show_porcupines_reference(state == Qt.Checked.value)

    def _on_toggle_deviation(self, state):
        """Affiche/masque la deviation entre profils."""
        self._canvas.set_show_deviation(
            state == Qt.Checked.value)

    def load_profil_from_file(self, filepath, role="current"):
        """Charge un profil depuis un fichier.

        :param filepath: chemin du fichier profil
        :type filepath: str
        :param role: 'current' ou 'reference'
        :type role: str
        :returns: (succes, nom_ou_message_erreur)
        :rtype: tuple(bool, str)
        """
        try:
            profil = ProfilSpline.from_file(filepath)
            profil.normalize()
        except Exception as e:
            return False, str(e)

        if role == "reference":
            self._profil_reference = profil
            self._lbl_reference.setText(profil.name)
            self._chk_reference.setChecked(True)
            self._canvas.set_reference_profil(self._profil_reference)
        else:
            self._profil_current = profil
            self._lbl_current.setText(profil.name)
            self._chk_current.setChecked(True)
            self._canvas.set_current_profil(self._profil_current)
        self.profil_changed.emit(role)
        return True, profil.name

    def convert_current_to_spline(self, degree_ext=6, degree_int=6,
                                   max_dev=0.001, max_segments=8,
                                   smoothing=0.0):
        """Convertit le profil courant en mode Spline.

        :param degree_ext: degre pour l'extrados
        :type degree_ext: int
        :param degree_int: degre pour l'intrados
        :type degree_int: int
        :param max_dev: deviation max toleree (pour mode adaptatif)
        :type max_dev: float
        :param max_segments: nombre max de segments par cote
        :type max_segments: int
        :param smoothing: poids de regularisation
        :type smoothing: float
        :returns: (True, nom) si ok, (False, message) si erreur,
                  (None, None) si pas de profil ou deja en Spline
        :rtype: tuple
        """
        p = self._profil_current
        if p is None:
            return None, None
        if p.has_splines:
            return None, None

        try:
            # Extrados
            p.approximate_spline(
                degree=degree_ext, max_dev=max_dev,
                smoothing=smoothing, max_segments=max_segments)
            # Ajuster le degre intrados si different
            if degree_int != degree_ext:
                for seg in p.spline_intrados._segments:
                    current_deg = seg.degree
                    if degree_int > current_deg:
                        seg.elevate(degree_int - current_deg)
                    elif degree_int < current_deg:
                        seg.reduce(current_deg - degree_int)
                p.spline_intrados._invalidate(geometry=True)
        except Exception as e:
            return False, str(e)

        self._canvas.set_current_profil(p)
        self.profil_changed.emit('current')
        return True, p.name

    def save_current_profil(self):
        """Sauvegarde le profil courant via un dialogue fichier.

        :returns: (None, None) si annule/pas de profil,
                  (True, chemin) si ok, (False, message) si erreur
        :rtype: tuple
        """
        if self._profil_current is None:
            return None, None

        from PySide6.QtWidgets import QFileDialog
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Sauvegarder le profil courant",
            "%s.dat" % self._profil_current.name,
            u"Selig (*.dat);;Lednicer (*.dat);;Spline (*.bspl);;CSV (*.csv)"
        )
        if not filepath:
            return None, None

        if "CSV" in selected_filter:
            fmt = 'csv'
        elif "Lednicer" in selected_filter:
            fmt = 'lednicer'
        elif "Spline" in selected_filter:
            fmt = 'bspl'
        else:
            fmt = 'selig'

        try:
            self._profil_current.write(filepath, fmt=fmt)
        except Exception as e:
            return False, str(e)

        return True, filepath

    @property
    def profil_current(self):
        """Retourne le profil courant (ou None)."""
        return self._profil_current

    @property
    def profil_reference(self):
        """Retourne le profil de reference (ou None)."""
        return self._profil_reference

    def change_sampling(self, role='current'):
        """Change le nombre de points d'echantillonnage d'un profil Bezier.

        :param role: 'current' ou 'reference'
        :type role: str
        :returns: (True, info), (False, erreur) ou (None, message)
        :rtype: tuple
        """
        profil = (self._profil_current if role == 'current'
                  else self._profil_reference)
        if profil is None:
            return None, "Pas de profil"
        if not profil.has_splines:
            return None, u"Le profil n'est pas en mode Spline"

        current_n = profil.spline_extrados.n_points
        from PySide6.QtWidgets import QInputDialog
        label = "courant" if role == 'current' else u"r\u00e9f\u00e9rence"
        value, ok = QInputDialog.getInt(
            self,
            u"\u00c9chantillonnage profil %s" % label,
            "Nombre de points :",
            current_n, 10, 10000, 10)
        if not ok:
            return None, None

        profil.spline_extrados.n_points = value
        profil.spline_intrados.n_points = value
        if role == 'current':
            self._canvas.set_current_profil(profil)
        else:
            self._canvas.set_reference_profil(profil)
        self.profil_changed.emit(role)
        return True, "%s : %d points" % (profil.name, value)

    def zoom_fit(self):
        """Zoom adapte (appele depuis le menu)."""
        self._canvas.zoom_fit()
