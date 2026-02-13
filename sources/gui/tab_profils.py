#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Onglet Profils : affichage et edition interactive des profils."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QLabel,
    QGroupBox, QSpinBox
)
from PySide6.QtCore import Qt

from model.profil import Profil
from .profil_canvas import ProfilCanvas


class TabProfils(QWidget):
    """Onglet principal d'edition des profils aerodynamiques."""

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

        ctrl_layout.addSpacing(20)

        # Degre Bezier
        ctrl_layout.addWidget(QLabel(u"Degr\u00e9 B\u00e9zier:"))
        self._spn_degree = QSpinBox()
        self._spn_degree.setRange(2, 15)
        self._spn_degree.setValue(5)
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

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # --- Canvas matplotlib ---
        self._canvas = ProfilCanvas(self)
        layout.addWidget(self._canvas, stretch=1)

    def _load_default_profiles(self):
        """Charge les profils par defaut au demarrage."""
        # Profil courant : NACA 2412 en mode Bezier
        self._profil_current = Profil.from_naca('2412', n_points=150)
        self._profil_current.normalize()
        degree = self._spn_degree.value()
        self._profil_current.approximate_bezier(degree=degree)

        # Profil reference : NACA 0012 en mode Bezier
        self._profil_reference = Profil.from_naca('0012', n_points=150)
        self._profil_reference.normalize()
        self._profil_reference.approximate_bezier(degree=degree)

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

    def _on_degree_changed(self, degree):
        """Change le degre des Beziers du profil courant."""
        if self._profil_current is not None:
            self._profil_current.clear_beziers()
            self._profil_current.approximate_bezier(degree=degree)
            self._canvas.set_current_profil(self._profil_current)

    def zoom_fit(self):
        """Zoom adapte (appele depuis le menu)."""
        self._canvas.zoom_fit()
