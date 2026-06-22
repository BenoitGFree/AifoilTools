#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Dialogue de diagnostic d'une simulation XFoil.

Donne acces au repertoire de travail d'une simulation et au contenu de
ses fichiers (log console XFoil, script de commandes, profil envoye,
polaires, Cp, couche limite) pour diagnostiquer un calcul.
"""

import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPlainTextEdit, QPushButton
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QTextCursor
from .i18n import tr as _


# Ordre de priorite d'affichage : le bilan Python d'abord (le plus
# parlant), puis le log console XFoil, le script, puis les sorties.
_PRIORITY = ('.cmd.log', '.log', '.cmd', '.dat', '.txt')


def _sort_key(name):
    u"""Cle de tri : diagnostic.log et logs en premier, puis alphabetique."""
    low = name.lower()
    if low == 'diagnostic.log':
        return (-1, low)
    for i, suffix in enumerate(_PRIORITY):
        if low.endswith(suffix):
            return (i, low)
    return (len(_PRIORITY), low)


class DiagnosticDialog(QDialog):
    u"""Fenetre de consultation des fichiers d'une simulation XFoil."""

    def __init__(self, work_dir, title=u"profil", parent=None):
        u"""
        :param work_dir: repertoire de travail de la simulation
        :type work_dir: str
        :param title: libelle du profil (courant, reference, volet)
        :type title: str
        """
        super().__init__(parent)
        self._work_dir = work_dir
        self.setWindowTitle(_(u"Diagnostic XFoil — %s") % title)
        self.resize(820, 600)
        self._build_ui()
        self._populate_files()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # En-tete : chemin du repertoire
        path_row = QHBoxLayout()
        lbl = QLabel(_(u"Dossier :"))
        path_row.addWidget(lbl)
        self._path_label = QLabel(self._work_dir or u"(indisponible)")
        self._path_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse)
        self._path_label.setStyleSheet("color: #555;")
        path_row.addWidget(self._path_label, stretch=1)
        self._btn_open = QPushButton(_(u"Ouvrir le dossier"))
        self._btn_open.setToolTip(
            _(u"Ouvre le repertoire de travail dans l'explorateur de"
            u" fichiers."))
        self._btn_open.clicked.connect(self._open_folder)
        path_row.addWidget(self._btn_open)
        layout.addLayout(path_row)

        # Selecteur de fichier
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel(_(u"Fichier :")))
        self._combo = QComboBox()
        self._combo.setToolTip(
            _(u"Choisissez le fichier a inspecter.\n"
            u"Le log console XFoil (.cmd.log) contient les messages de"
            u" convergence et d'erreur."))
        self._combo.currentIndexChanged.connect(self._show_selected)
        file_row.addWidget(self._combo, stretch=1)
        self._btn_refresh = QPushButton(_(u"Rafraîchir"))
        self._btn_refresh.clicked.connect(self._populate_files)
        file_row.addWidget(self._btn_refresh)
        layout.addLayout(file_row)

        # Zone de texte (monospace)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.NoWrap)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(9)
        self._text.setFont(mono)
        layout.addWidget(self._text, stretch=1)

        # Bouton fermer
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_close = QPushButton(_(u"Fermer"))
        self._btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_close)
        layout.addLayout(btn_row)

    def _populate_files(self):
        u"""Liste les fichiers du repertoire de travail."""
        self._combo.blockSignals(True)
        self._combo.clear()
        files = []
        if self._work_dir and os.path.isdir(self._work_dir):
            for name in os.listdir(self._work_dir):
                full = os.path.join(self._work_dir, name)
                if os.path.isfile(full):
                    files.append(name)
        files.sort(key=_sort_key)
        for name in files:
            self._combo.addItem(name)
        self._combo.blockSignals(False)

        has_files = self._combo.count() > 0
        self._btn_open.setEnabled(
            bool(self._work_dir) and os.path.isdir(self._work_dir or ''))
        if has_files:
            self._combo.setCurrentIndex(0)
            self._show_selected()
        else:
            self._text.setPlainText(
                u"Aucun fichier de simulation disponible.\n\n"
                u"Lancez une simulation pour ce profil depuis l'onglet"
                u" Paramétrage XFoil.")

    def _show_selected(self):
        u"""Affiche le contenu du fichier selectionne."""
        name = self._combo.currentText()
        if not name or not self._work_dir:
            return
        full = os.path.join(self._work_dir, name)
        try:
            with open(full, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except OSError as e:
            content = u"Impossible de lire le fichier :\n%s" % str(e)
        if not content.strip():
            content = u"(fichier vide)"
        self._text.setPlainText(content)
        # Aller a la fin pour les logs (messages recents en bas)
        if name.lower().endswith('.log'):
            self._text.moveCursor(QTextCursor.End)

    def _open_folder(self):
        u"""Ouvre le repertoire de travail dans l'explorateur."""
        if self._work_dir and os.path.isdir(self._work_dir):
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(self._work_dir))
