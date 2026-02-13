#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Onglet Parametrage XFoil (placeholder)."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class TabXfoil(QWidget):
    """Onglet de parametrage des simulations XFoil."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(u"Param\u00e9trage XFoil \u2014 \u00e0 venir")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
