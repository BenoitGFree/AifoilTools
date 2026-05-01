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

        act_open_current = QAction("Ouvrir profil &courant...", self)
        act_open_current.setShortcut("Ctrl+O")
        act_open_current.triggered.connect(self._on_open_current)
        file_menu.addAction(act_open_current)

        act_open_ref = QAction(u"Ouvrir profil r\u00e9f\u00e9rence...", self)
        act_open_ref.setShortcut("Ctrl+Shift+O")
        act_open_ref.triggered.connect(self._on_open_reference)
        file_menu.addAction(act_open_ref)

        file_menu.addSeparator()

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

        edit_menu.addSeparator()

        act_to_spline = QAction("Convertir en Spline", self)
        act_to_spline.setShortcut("Ctrl+B")
        act_to_spline.triggered.connect(self._on_convert_to_spline)
        edit_menu.addAction(act_to_spline)

        edit_menu.addSeparator()

        sample_menu = edit_menu.addMenu(u"\u00c9chantillonnage")
        act_sample_cur = sample_menu.addAction("Profil &courant...")
        act_sample_cur.triggered.connect(
            lambda: self._on_change_sampling('current'))
        act_sample_ref = sample_menu.addAction(
            u"Profil r\u00e9f\u00e9rence...")
        act_sample_ref.triggered.connect(
            lambda: self._on_change_sampling('reference'))

        # --- Affichage ---
        view_menu = menubar.addMenu("&Affichage")

        act_zoom_fit = QAction("Zoom &adapte", self)
        act_zoom_fit.setShortcut("Ctrl+0")
        act_zoom_fit.triggered.connect(self._on_zoom_fit)
        view_menu.addAction(act_zoom_fit)

        # Sous-menu Disposition
        disp_menu = view_menu.addMenu("&Disposition")
        self._disp_actions = []
        for rows, cols in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]:
            label = "%d \u00d7 %d" % (rows, cols)
            act = QAction(label, self)
            act.setCheckable(True)
            if rows == 2 and cols == 2:
                act.setChecked(True)
            act.triggered.connect(
                lambda checked, r=rows, c=cols: self._on_set_grid(r, c))
            disp_menu.addAction(act)
            self._disp_actions.append((rows, cols, act))

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

        # Connecter le bouton Lancer
        self._tab_xfoil.run_requested.connect(self._on_run_simulations)

        # Marquer les resultats obsoletes quand un profil change
        self._tab_profils.profil_changed.connect(
            self._tab_results.mark_stale)

        # Worker en cours
        self._sim_worker = None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _open_profil(self, role):
        """Ouvre un fichier profil et le charge comme courant ou reference."""
        from PySide6.QtWidgets import QFileDialog
        label = "courant" if role == "current" else u"r\u00e9f\u00e9rence"
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir profil %s" % label,
            "",
            u"Profils (*.dat);;Spline (*.bspl);;B\u00e9zier (*.bez)"
            u";;CSV (*.csv);;Tous (*)"
        )
        if not filepath:
            return

        ok, info = self._tab_profils.load_profil_from_file(filepath, role)
        if ok:
            self.statusBar().showMessage("Profil %s charge : %s" % (label, info))
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Erreur de chargement", info)
            self.statusBar().showMessage("Echec du chargement")

    def _on_open_current(self):
        """Ouvre un fichier comme profil courant."""
        self._open_profil("current")

    def _on_open_reference(self):
        """Ouvre un fichier comme profil de reference."""
        self._open_profil("reference")

    def _on_save(self):
        """Sauvegarde le profil courant dans un fichier."""
        ok, info = self._tab_profils.save_current_profil()
        if ok is None:
            return  # annule ou pas de profil
        if ok:
            self.statusBar().showMessage("Profil sauvegarde : %s" % info)
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Erreur de sauvegarde", info)
            self.statusBar().showMessage("Echec de la sauvegarde")

    def _on_change_sampling(self, role):
        """Change le nombre de points d'echantillonnage d'un profil Bezier."""
        ok, info = self._tab_profils.change_sampling(role)
        if ok is None:
            self.statusBar().showMessage(info or "")
        elif ok:
            self.statusBar().showMessage(
                u"\u00c9chantillonnage : %s" % info)
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, u"\u00c9chantillonnage", info)

    def _on_convert_to_spline(self):
        """Convertit le profil courant en mode Spline."""
        p = self._tab_profils.profil_current
        if p is None or p.has_splines:
            self.statusBar().showMessage(
                u"Pas de profil ou d\u00e9j\u00e0 en mode Spline")
            return

        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QFormLayout, QSpinBox,
            QDoubleSpinBox
        )
        dlg = QDialog(self)
        dlg.setWindowTitle("Convertir en Spline")
        form = QFormLayout(dlg)

        spn_ext = QSpinBox()
        spn_ext.setRange(2, 30)
        spn_ext.setValue(11)
        form.addRow(u"Degr\u00e9 extrados :", spn_ext)

        spn_int = QSpinBox()
        spn_int.setRange(2, 30)
        spn_int.setValue(11)
        form.addRow(u"Degr\u00e9 intrados :", spn_int)

        spn_tol = QDoubleSpinBox()
        spn_tol.setRange(0.0001, 0.1)
        spn_tol.setDecimals(4)
        spn_tol.setSingleStep(0.0005)
        spn_tol.setValue(0.001)
        spn_tol.setToolTip(
            u"D\u00e9viation max acceptable (corde = 1).\n"
            u"Plus petit = plus de segments.")
        form.addRow(u"Tol\u00e9rance :", spn_tol)

        spn_max_seg = QSpinBox()
        spn_max_seg.setRange(1, 20)
        spn_max_seg.setValue(1)
        spn_max_seg.setToolTip(
            u"Nombre max de segments B\u00e9zier par c\u00f4t\u00e9.\n"
            u"1 = un seul segment (ancien comportement).")
        form.addRow("Max segments :", spn_max_seg)

        spn_smooth = QDoubleSpinBox()
        spn_smooth.setRange(0.0, 1.0)
        spn_smooth.setDecimals(2)
        spn_smooth.setSingleStep(0.01)
        spn_smooth.setValue(0.1)
        spn_smooth.setToolTip(
            u"R\u00e9gularisation (0 = pas de lissage).\n"
            u"Valeurs > 0 lissent le polygone de contr\u00f4le.")
        form.addRow("Lissage :", spn_smooth)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() != QDialog.Accepted:
            return

        ok, info = self._tab_profils.convert_current_to_spline(
            degree_ext=spn_ext.value(),
            degree_int=spn_int.value(),
            max_dev=spn_tol.value(),
            max_segments=spn_max_seg.value(),
            smoothing=spn_smooth.value())
        if ok is None:
            self.statusBar().showMessage(
                u"Pas de profil ou d\u00e9j\u00e0 en mode Spline")
        elif ok:
            self.statusBar().showMessage(
                "Profil '%s' converti en Spline" % info)
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Erreur de conversion", info)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _on_run_simulations(self, target='both'):
        u"""Lance les simulations XFoil pour les profils demandes.

        :param target: 'both', 'current' ou 'reference'
        :type target: str
        """
        if self._sim_worker is not None and self._sim_worker.isRunning():
            self.statusBar().showMessage("Simulation deja en cours...")
            return

        profils = {}
        if target in ('both', 'current'):
            profils['current'] = self._tab_profils.profil_current
        if target in ('both', 'reference'):
            profils['reference'] = self._tab_profils.profil_reference

        # Retirer les None
        profils = {k: v for k, v in profils.items() if v is not None}
        if not profils:
            self.statusBar().showMessage("Aucun profil a simuler")
            return

        self._sim_target = target
        params = self._tab_xfoil.get_params()

        from .simulation_worker import SimulationWorker
        self._sim_worker = SimulationWorker(profils, params, parent=self)
        self._sim_worker.progress.connect(self._on_sim_progress)
        self._sim_worker.finished_ok.connect(self._on_sim_finished)
        self._sim_worker.finished_error.connect(self._on_sim_error)

        self._tab_xfoil.set_enabled(False)
        self.statusBar().showMessage("Simulations en cours...")
        self._sim_worker.start()

    def _on_sim_progress(self, msg):
        """Met a jour la barre de statut pendant la simulation."""
        self.statusBar().showMessage(msg)

    def _on_sim_finished(self, results):
        u"""Traite les resultats des simulations."""
        self._tab_xfoil.set_enabled(True)
        self._sim_worker = None

        n_profils = len(results)
        total_pts = sum(r.n_converged for r in results.values())
        self.statusBar().showMessage(
            "Terminees : %d profil(s), %d pts converges"
            % (n_profils, total_pts))

        # Merge partiel ou remplacement complet
        target = getattr(self, '_sim_target', 'both')
        if target == 'both':
            self._tab_results.set_results(results)
        else:
            self._tab_results.update_results(results)

        # Basculer sur l'onglet Resultats
        self._tabs.setCurrentWidget(self._tab_results)

    def _on_sim_error(self, error_msg):
        """Affiche l'erreur de simulation."""
        self._tab_xfoil.set_enabled(True)
        self._sim_worker = None

        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Erreur de simulation", error_msg)
        self.statusBar().showMessage("Echec de la simulation")

    def _on_set_grid(self, rows, cols):
        """Change la disposition de la grille de resultats."""
        self._tab_results.set_grid(rows, cols)
        # Mettre a jour les coches du menu
        for r, c, act in self._disp_actions:
            act.setChecked(r == rows and c == cols)

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
