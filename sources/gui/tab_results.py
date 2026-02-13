#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Onglet Resultats : grille configurable de cellules d'analyse.

L'onglet affiche une grille NxM de ResultCell, chacune avec :
- un combo pour choisir l'analyse (CL, CD, finesse, etc.)
- un canvas matplotlib interactif (zoom, pan, coordonnees curseur)
- des checkboxes pour afficher courant / reference
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel
from PySide6.QtCore import Qt

from .result_cell import ResultCell, DEFAULT_ANALYSES

_STALE_STYLE = (
    "background-color: #fff3cd; color: #856404; "
    "border: 1px solid #ffc107; border-radius: 3px; padding: 4px;"
)

_ROLE_LABELS = {
    'current': 'courant',
    'reference': u'r\u00e9f\u00e9rence',
}


class TabResults(QWidget):
    u"""Onglet de presentation des resultats de simulation.

    Gere une grille configurable de cellules d'analyse independantes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = {}      # {'current': SimulationResults, ...}
        self._n_rows = 2
        self._n_cols = 2
        self._cells = []        # liste plate de ResultCell
        self._stale_roles = set()  # roles modifies depuis la simulation

        self._build_ui()
        self._rebuild_grid()

    def _build_ui(self):
        u"""Construit l'interface de l'onglet."""
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(4, 4, 4, 4)

        # Bandeau "resultats obsoletes"
        self._stale_label = QLabel()
        self._stale_label.setAlignment(Qt.AlignCenter)
        self._stale_label.setStyleSheet(_STALE_STYLE)
        self._stale_label.setWordWrap(True)
        self._stale_label.setVisible(False)
        self._main_layout.addWidget(self._stale_label)

        # Label resume
        self._label = QLabel("Aucun resultat. Lancez une simulation.")
        self._label.setAlignment(Qt.AlignCenter)
        self._main_layout.addWidget(self._label)

        # Conteneur pour la grille
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(2)
        self._main_layout.addWidget(self._grid_widget, stretch=1)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def set_results(self, results):
        u"""Affiche les resultats des simulations (remplacement complet).

        :param results: dict {'current': SimulationResults, 'reference': ...}
        :type results: dict
        """
        self._results = results
        self._stale_roles.clear()
        self._update_stale_label()
        self._update_label()
        for cell in self._cells:
            cell.set_results(results)

    def update_results(self, new_results):
        u"""Met a jour les resultats partiellement (merge avec les existants).

        Utile pour relancer la simulation d'un seul profil sans perdre
        les resultats de l'autre.

        :param new_results: dict {'current': SimulationResults} ou similaire
        :type new_results: dict
        """
        self._results.update(new_results)
        for role in new_results:
            self._stale_roles.discard(role)
        self._update_stale_label()
        self._update_label()
        for cell in self._cells:
            cell.set_results(self._results)

    def mark_stale(self, role):
        u"""Marque un profil comme modifie depuis la simulation.

        :param role: 'current' ou 'reference'
        :type role: str
        """
        if not self._results:
            return
        self._stale_roles.add(role)
        self._update_stale_label()

    def set_grid(self, n_rows, n_cols):
        u"""Change la disposition de la grille.

        :param n_rows: nombre de lignes
        :param n_cols: nombre de colonnes
        """
        if n_rows == self._n_rows and n_cols == self._n_cols:
            return
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._rebuild_grid()

    @property
    def grid_size(self):
        """Retourne (n_rows, n_cols)."""
        return self._n_rows, self._n_cols

    # ------------------------------------------------------------------
    # Construction de la grille
    # ------------------------------------------------------------------

    def _rebuild_grid(self):
        u"""Reconstruit la grille de cellules."""
        # Memoriser les analyses en cours (si elles existent)
        old_analyses = [c.analysis_name for c in self._cells]

        # Supprimer les anciennes cellules
        for cell in self._cells:
            self._grid_layout.removeWidget(cell)
            cell.deleteLater()
        self._cells = []

        # Creer les nouvelles cellules
        n_total = self._n_rows * self._n_cols
        for idx in range(n_total):
            # Choisir l'analyse : reprendre l'ancienne si possible,
            # sinon utiliser les defauts cycliquement
            if idx < len(old_analyses):
                analysis = old_analyses[idx]
            else:
                analysis = DEFAULT_ANALYSES[idx % len(DEFAULT_ANALYSES)]

            cell = ResultCell(analysis_name=analysis, parent=self)
            if self._results:
                cell.set_results(self._results)

            row = idx // self._n_cols
            col = idx % self._n_cols
            self._grid_layout.addWidget(cell, row, col)
            self._cells.append(cell)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def _update_stale_label(self):
        u"""Met a jour le bandeau d'avertissement."""
        if not self._stale_roles:
            self._stale_label.setVisible(False)
            return
        names = [_ROLE_LABELS.get(r, r) for r in sorted(self._stale_roles)]
        profils_txt = " et ".join(names)
        self._stale_label.setText(
            u"\u26a0 Le profil %s a \u00e9t\u00e9 modifi\u00e9 "
            u"depuis la derni\u00e8re simulation. "
            u"Relancez pour mettre \u00e0 jour." % profils_txt)
        self._stale_label.setVisible(True)

    def _update_label(self):
        u"""Met a jour le label de resume."""
        if not self._results:
            self._label.setText("Aucun resultat.")
            return

        summaries = []
        for role, sim_results in self._results.items():
            label = "Courant" if role == 'current' \
                else u"R\u00e9f\u00e9rence"
            n = sim_results.n_converged
            n_re = len(sim_results.re_list)
            summaries.append(
                "%s : %d Re, %d pts" % (label, n_re, n))
        self._label.setText(" | ".join(summaries))
