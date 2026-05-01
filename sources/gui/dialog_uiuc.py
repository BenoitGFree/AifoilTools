#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Dialogue de chargement d'un profil depuis la base UIUC.

Permet de rechercher un profil dans l'index UIUC, le telecharger
(avec mise en cache) et le retourner pour chargement comme profil
courant ou de reference.
"""

import logging

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QLabel, QPushButton, QDialogButtonBox,
    QProgressDialog, QMessageBox, QApplication
)
from PySide6.QtCore import Qt, Signal, QThread

from model.uiuc_loader import UIUCLoader

logger = logging.getLogger(__name__)


class _IndexFetcher(QThread):
    u"""Worker pour telecharger l'index sans bloquer la GUI."""

    finished_ok = Signal(list)        # liste d'UIUCEntry
    finished_error = Signal(str)

    def __init__(self, loader, force=False, parent=None):
        super().__init__(parent)
        self._loader = loader
        self._force = force

    def run(self):
        try:
            entries = self._loader.fetch_index(force=self._force)
            self.finished_ok.emit(entries)
        except Exception as exc:
            logger.exception("Echec fetch_index")
            self.finished_error.emit(str(exc))


class _DatFetcher(QThread):
    u"""Worker pour telecharger un fichier .dat sans bloquer la GUI."""

    finished_ok = Signal(str)         # chemin local
    finished_error = Signal(str)

    def __init__(self, loader, entry, parent=None):
        super().__init__(parent)
        self._loader = loader
        self._entry = entry

    def run(self):
        try:
            path = self._loader.fetch_dat(self._entry)
            self.finished_ok.emit(str(path))
        except Exception as exc:
            logger.exception("Echec fetch_dat")
            self.finished_error.emit(str(exc))


class DialogUIUC(QDialog):
    u"""Dialogue de selection d'un profil dans la base UIUC.

    Apres acceptation, le chemin local du .dat est disponible via
    :attr:`selected_path`. Utiliser :class:`exec()` puis tester
    ``result() == QDialog.Accepted``.

    :param parent: widget parent
    :param role_label: chaine indiquant le role ("courant" ou "reference")
    :type role_label: str
    """

    def __init__(self, parent=None, role_label=u"courant"):
        super().__init__(parent)
        self.setWindowTitle(
            u"Charger un profil UIUC (%s)" % role_label)
        self.resize(560, 460)

        self._loader = UIUCLoader()
        self._entries = []          # liste d'UIUCEntry
        self._selected_entry = None
        self.selected_path = None

        self._build_ui()
        # Telecharger l'index au demarrage (depuis le cache si possible)
        self._refresh_index(force=False)

    # ------------------------------------------------------------------
    #  UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Ligne de recherche
        line = QHBoxLayout()
        line.addWidget(QLabel(u"Recherche :"))
        self._edit_search = QLineEdit()
        self._edit_search.setPlaceholderText(
            u"naca, eppler, fx, sd...")
        self._edit_search.textChanged.connect(self._on_filter_changed)
        self._edit_search.setToolTip(
            u"Filtre la liste des profils par nom ou description.\n"
            u"La recherche est insensible a la casse.")
        line.addWidget(self._edit_search, 1)

        self._btn_refresh = QPushButton(u"Rafraichir")
        self._btn_refresh.setToolTip(
            u"Re-telecharge l'index UIUC depuis le serveur (ignore"
            u" le cache local).")
        self._btn_refresh.clicked.connect(
            lambda: self._refresh_index(force=True))
        line.addWidget(self._btn_refresh)
        layout.addLayout(line)

        # Liste des profils
        self._list = QListWidget()
        self._list.setToolTip(
            u"Liste des profils UIUC. Cliquer pour selectionner ;"
            u" double-cliquer pour charger directement.")
        self._list.currentItemChanged.connect(self._on_selection)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list, 1)

        # Description du profil selectionne
        self._lbl_desc = QLabel(u"Aucun profil selectionne")
        self._lbl_desc.setWordWrap(True)
        self._lbl_desc.setStyleSheet(
            "color: #555; font-style: italic; padding: 4px;")
        layout.addWidget(self._lbl_desc)

        # Compteur
        self._lbl_count = QLabel(u"")
        self._lbl_count.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self._lbl_count)

        # Boutons OK / Annuler
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.button(QDialogButtonBox.Ok).setText(u"Charger")
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    # ------------------------------------------------------------------
    #  Index
    # ------------------------------------------------------------------

    def _refresh_index(self, force):
        u"""Lance le telechargement de l'index dans un worker."""
        self._set_busy(True, u"Chargement de l'index UIUC...")
        self._index_worker = _IndexFetcher(
            self._loader, force=force, parent=self)
        self._index_worker.finished_ok.connect(self._on_index_ok)
        self._index_worker.finished_error.connect(self._on_index_error)
        self._index_worker.start()

    def _on_index_ok(self, entries):
        self._entries = entries
        self._populate_list(entries)
        self._set_busy(False)
        self._lbl_count.setText(
            u"%d profils dans l'index" % len(entries))

    def _on_index_error(self, msg):
        self._set_busy(False)
        QMessageBox.warning(
            self, u"Erreur de telechargement",
            u"Impossible de telecharger l'index UIUC.\n\n%s\n\n"
            u"Verifiez votre connexion internet ou utilisez un\n"
            u"profil deja present dans le cache local."
            % msg)

    def _populate_list(self, entries):
        self._list.clear()
        for e in entries:
            item = QListWidgetItem("%s   %s" % (
                e.name.ljust(22), e.description))
            item.setData(Qt.UserRole, e)
            self._list.addItem(item)

    def _on_filter_changed(self, text):
        u"""Filtre la liste selon le texte de recherche."""
        text = text.strip().lower()
        n_visible = 0
        for i in range(self._list.count()):
            item = self._list.item(i)
            entry = item.data(Qt.UserRole)
            haystack = (entry.name + ' ' + entry.description).lower()
            visible = (not text) or (text in haystack)
            item.setHidden(not visible)
            if visible:
                n_visible += 1
        if self._entries:
            self._lbl_count.setText(
                u"%d / %d profils visibles"
                % (n_visible, len(self._entries)))

    # ------------------------------------------------------------------
    #  Selection
    # ------------------------------------------------------------------

    def _on_selection(self, current, _previous):
        if current is None:
            self._selected_entry = None
            self._lbl_desc.setText(u"Aucun profil selectionne")
            self._buttons.button(
                QDialogButtonBox.Ok).setEnabled(False)
            return
        self._selected_entry = current.data(Qt.UserRole)
        self._lbl_desc.setText(self._selected_entry.description
                               or u"(pas de description)")
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(True)

    def _on_double_click(self, _item):
        self._on_accept()

    def _on_accept(self):
        if self._selected_entry is None:
            return
        self._set_busy(True, u"Telechargement de %s..."
                              % self._selected_entry.name)
        self._dat_worker = _DatFetcher(
            self._loader, self._selected_entry, parent=self)
        self._dat_worker.finished_ok.connect(self._on_dat_ok)
        self._dat_worker.finished_error.connect(self._on_dat_error)
        self._dat_worker.start()

    def _on_dat_ok(self, path):
        self._set_busy(False)
        self.selected_path = path
        self.accept()

    def _on_dat_error(self, msg):
        self._set_busy(False)
        QMessageBox.warning(
            self, u"Erreur de telechargement",
            u"Impossible de telecharger le profil %s.\n\n%s"
            % (self._selected_entry.name, msg))

    # ------------------------------------------------------------------
    #  Etat occupe
    # ------------------------------------------------------------------

    def _set_busy(self, busy, message=u""):
        u"""Active/desactive l'UI pendant une operation reseau."""
        self._edit_search.setEnabled(not busy)
        self._btn_refresh.setEnabled(not busy)
        self._list.setEnabled(not busy)
        self._buttons.setEnabled(not busy)
        if busy:
            self._lbl_count.setText(message)
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
