#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fenetre principale AifoilTools."""

import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QMenuBar, QStatusBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from .tab_profils import TabProfils
from .tab_xfoil import TabXfoil
from .tab_results import TabResults


class MainWindow(QMainWindow):
    """Fenetre principale avec menu et onglets."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AifoilTools")
        self.resize(1200, 700)

        self._build_menus()
        self._build_tabs()
        self.statusBar().showMessage("Pret")

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _build_menus(self):
        """Construit la barre de menus."""
        menubar = self.menuBar()

        # --- Fichier ---
        file_menu = menubar.addMenu("&Fichier")

        act_open = QAction("&Ouvrir profil...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open)
        file_menu.addAction(act_open)

        act_save = QAction("&Sauvegarder profil...", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._on_save)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_quit = QAction("&Quitter", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # --- Edition ---
        edit_menu = menubar.addMenu("&Edition")

        act_undo = QAction("&Annuler", self)
        act_undo.setShortcut("Ctrl+Z")
        edit_menu.addAction(act_undo)

        act_redo = QAction("&Refaire", self)
        act_redo.setShortcut("Ctrl+Y")
        edit_menu.addAction(act_redo)

        # --- Affichage ---
        view_menu = menubar.addMenu("&Affichage")

        act_zoom_fit = QAction("Zoom &adapte", self)
        act_zoom_fit.setShortcut("Ctrl+0")
        act_zoom_fit.triggered.connect(self._on_zoom_fit)
        view_menu.addAction(act_zoom_fit)

        # --- Aide ---
        help_menu = menubar.addMenu("&Aide")

        act_about = QAction("A &propos...", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    # ------------------------------------------------------------------
    # Onglets
    # ------------------------------------------------------------------

    def _build_tabs(self):
        """Construit le widget a onglets."""
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        self._tab_profils = TabProfils()
        self._tab_xfoil = TabXfoil()
        self._tab_results = TabResults()

        self._tabs.addTab(self._tab_profils, "Profils")
        self._tabs.addTab(self._tab_xfoil, u"Param\u00e9trage XFoil")
        self._tabs.addTab(self._tab_results, u"R\u00e9sultats")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_open(self):
        """Ouvre un fichier profil."""
        self.statusBar().showMessage("Ouvrir profil... (a implementer)")

    def _on_save(self):
        """Sauvegarde le profil courant."""
        self.statusBar().showMessage("Sauvegarder... (a implementer)")

    def _on_zoom_fit(self):
        """Zoom adapte sur le canvas."""
        self._tab_profils.zoom_fit()

    def _on_about(self):
        """Affiche la boite A propos."""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self, "AifoilTools",
            "AifoilTools - Analyse aerodynamique 2D\n\n"
            "Courbes de Bezier, profils, XFoil\n"
            u"\u00a9 Nervures"
        )


def main():
    """Point d'entree de l'application."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
