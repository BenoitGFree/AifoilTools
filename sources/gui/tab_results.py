#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Onglet Resultats (placeholder)."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class TabResults(QWidget):
    """Onglet de presentation des resultats de simulation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(u"R\u00e9sultats \u2014 \u00e0 venir")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
