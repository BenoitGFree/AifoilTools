#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Onglet Profils : affichage et edition interactive des profils."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QLabel,
    QGroupBox, QSpinBox
)
from PySide6.QtCore import Qt, Signal

from model.profil import Profil
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

        ctrl_layout.addSpacing(20)

        # Degre Bezier
        ctrl_layout.addWidget(QLabel(u"Degr\u00e9 B\u00e9zier:"))
        self._spn_degree = QSpinBox()
        self._spn_degree.setRange(2, 15)
        self._spn_degree.setValue(6)
        self._spn_degree.valueChanged.connect(self._on_degree_changed)
        ctrl_layout.addWidget(self._spn_degree)

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
        self._profil_current = Profil.from_naca('2412', n_points=150)
        self._profil_current.normalize()

        self._profil_reference = Profil.from_naca('0012', n_points=150)
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

    def _on_toggle_porc_reference(self, state):
        """Affiche/masque les porcupines du profil de reference."""
        self._canvas.set_show_porcupines_reference(state == Qt.Checked.value)

    def _on_toggle_deviation(self, state):
        """Affiche/masque la deviation entre profils."""
        self._canvas.set_show_deviation(
            state == Qt.Checked.value)

    def _on_degree_changed(self, degree):
        """Change le degre des Beziers du profil courant (si en mode Bezier)."""
        if (self._profil_current is not None
                and self._profil_current.has_beziers):
            self._profil_current.clear_beziers()
            self._profil_current.approximate_bezier(degree=degree,
                                                    smoothing=0.1)
            self._canvas.set_current_profil(self._profil_current)
            self.profil_changed.emit('current')

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
            profil = Profil.from_file(filepath)
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

    def convert_current_to_bezier(self):
        """Convertit le profil courant en mode Bezier.

        :returns: (True, nom) si ok, (False, message) si erreur,
                  (None, None) si pas de profil ou deja en Bezier
        :rtype: tuple
        """
        p = self._profil_current
        if p is None:
            return None, None
        if p.has_beziers:
            return None, None

        try:
            degree = self._spn_degree.value()
            p.approximate_bezier(degree=degree, smoothing=0.1)
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
            u"Selig (*.dat);;Lednicer (*.dat);;B\u00e9zier (*.bez);;CSV (*.csv)"
        )
        if not filepath:
            return None, None

        if "CSV" in selected_filter:
            fmt = 'csv'
        elif "Lednicer" in selected_filter:
            fmt = 'lednicer'
        elif "zier" in selected_filter:
            fmt = 'bezier'
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

    def zoom_fit(self):
        """Zoom adapte (appele depuis le menu)."""
        self._canvas.zoom_fit()
