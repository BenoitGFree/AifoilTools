#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fenetre principale AirfoilTools."""

import os
import platform
import subprocess
import sys

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QMenuBar, QStatusBar
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

from . import __version__
from . import i18n
from .i18n import tr as _
from .tab_profils import TabProfils
from .tab_xfoil import TabXfoil
from .tab_results import TabResults
from .tab_cp import TabCp


def _find_manuel_pdf(lang='fr'):
    u"""Localise le manuel PDF pour la langue donnee (dev ou frozen).

    En anglais, cherche d'abord ``manuel_en.pdf`` ; a defaut (manuel
    anglais non encore disponible), retombe sur le manuel francais
    ``manuel.pdf``. En mode developpement les PDF sont dans
    ``<racine>/docs/manuel/`` ; en mode frozen (PyInstaller) dans
    ``<MEIPASS>/docs/``.

    :param lang: code de langue ('fr' ou 'en')
    :returns: chemin absolu du PDF, ou None si introuvable
    :rtype: str or None
    """
    names = ['manuel_en.pdf', 'manuel.pdf'] if lang == 'en' else ['manuel.pdf']
    if getattr(sys, 'frozen', False):
        roots = [
            os.path.join(sys._MEIPASS, 'docs'),
            os.path.join(os.path.dirname(sys.executable), 'docs'),
        ]
    else:
        # Mode dev : remonter de sources/gui/ vers la racine du projet
        base = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(
            os.path.join(base, '..', '..'))
        roots = [os.path.join(project_root, 'docs', 'manuel')]
    for name in names:
        for root in roots:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                return path
    return None


def _open_with_default_app(path):
    u"""Ouvre un fichier avec le visualiseur par defaut du systeme.

    :param path: chemin du fichier a ouvrir
    :type path: str
    """
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class MainWindow(QMainWindow):
    """Fenetre principale avec menu et onglets."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("AirfoilTools"))
        self.resize(1200, 700)

        # Backend de calcul ('xfoil' par defaut, 'flexfoil' si installe),
        # persiste via QSettings. Verifie face aux solveurs reellement
        # disponibles (FlexFoil ne l'est que si la dependance est presente).
        from model.pipeline import available_solvers
        self._available_solvers = available_solvers()
        from PySide6.QtCore import QSettings
        saved = QSettings().value('solver', 'xfoil')
        self._solver = (saved if saved in self._available_solvers
                        else 'xfoil')

        self._build_menus()
        self._build_tabs()
        self.statusBar().showMessage(_("Pret"))

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _build_menus(self):
        """Construit la barre de menus."""
        menubar = self.menuBar()

        # --- Fichier ---
        file_menu = menubar.addMenu(_("&Fichier"))

        act_open_current = QAction(_("Ouvrir profil &courant..."), self)
        act_open_current.setShortcut("Ctrl+O")
        act_open_current.setStatusTip(
            _(u"Charger un profil (.dat / .bspl / .bez / .csv) comme profil"
            u" courant (bleu)"))
        act_open_current.triggered.connect(self._on_open_current)
        file_menu.addAction(act_open_current)

        act_open_ref = QAction(_(u"Ouvrir profil r\u00e9f\u00e9rence..."), self)
        act_open_ref.setShortcut("Ctrl+Shift+O")
        act_open_ref.setStatusTip(
            _(u"Charger un profil comme profil de reference (rouge)"))
        act_open_ref.triggered.connect(self._on_open_reference)
        file_menu.addAction(act_open_ref)

        # Sous-menu UIUC (base de profils en ligne)
        uiuc_menu = file_menu.addMenu(_(u"Depuis la base &UIUC"))
        uiuc_menu.setStatusTip(
            _(u"Charger un profil depuis la base de donnees UIUC (Selig)"))
        act_uiuc_current = uiuc_menu.addAction(
            _(u"Profil courant\u2026"))
        act_uiuc_current.setStatusTip(
            _(u"Telecharger un profil depuis UIUC comme profil courant"))
        act_uiuc_current.triggered.connect(
            lambda: self._on_open_uiuc('current'))
        act_uiuc_ref = uiuc_menu.addAction(
            _(u"Profil r\u00e9f\u00e9rence\u2026"))
        act_uiuc_ref.setStatusTip(
            _(u"Telecharger un profil depuis UIUC comme profil de reference"))
        act_uiuc_ref.triggered.connect(
            lambda: self._on_open_uiuc('reference'))

        # Sous-menu NACA (generation a partir des indices)
        naca_menu = file_menu.addMenu(_(u"Profil &NACA"))
        naca_menu.setStatusTip(
            _(u"Generer un profil NACA (4 ou 5 chiffres) a partir de ses"
            u" indices"))
        act_naca_current = naca_menu.addAction(_(u"Profil courant…"))
        act_naca_current.setStatusTip(
            _(u"Generer un profil NACA comme profil courant"))
        act_naca_current.triggered.connect(
            lambda: self._on_new_naca('current'))
        act_naca_ref = naca_menu.addAction(_(u"Profil référence…"))
        act_naca_ref.setStatusTip(
            _(u"Generer un profil NACA comme profil de reference"))
        act_naca_ref.triggered.connect(
            lambda: self._on_new_naca('reference'))

        file_menu.addSeparator()

        act_save = QAction(_("&Sauvegarder profil..."), self)
        act_save.setShortcut("Ctrl+S")
        act_save.setStatusTip(
            _(u"Sauvegarder le profil courant (formats : Selig .dat,"
            u" Lednicer .dat, Spline .bspl, CSV)"))
        act_save.triggered.connect(self._on_save)
        file_menu.addAction(act_save)

        act_save_flap = QAction(_(u"Sauvegarder profil avec &volet..."), self)
        act_save_flap.setStatusTip(
            _(u"Sauvegarder le profil avec volet braque (vert), de la meme"
            u" maniere que le profil courant (complet, extrados ou"
            u" intrados). Necessite le volet active."))
        act_save_flap.triggered.connect(self._on_save_flap)
        file_menu.addAction(act_save_flap)

        file_menu.addSeparator()

        # --- Projet (.aftproj) ---
        act_open_project = QAction(_(u"Ouvrir &projet..."), self)
        act_open_project.setShortcut("Ctrl+Shift+P")
        act_open_project.setStatusTip(
            _(u"Ouvrir un projet AirfoilTools (.aftproj) : profils +"
            u" image de calque"))
        act_open_project.triggered.connect(self._on_open_project)
        file_menu.addAction(act_open_project)

        act_save_project = QAction(_(u"&Enregistrer projet..."), self)
        act_save_project.setShortcut("Ctrl+Alt+S")
        act_save_project.setStatusTip(
            _(u"Enregistrer le projet (profils courant/reference +"
            u" image de calque) au format .aftproj"))
        act_save_project.triggered.connect(self._on_save_project)
        file_menu.addAction(act_save_project)

        file_menu.addSeparator()

        # --- Image de calque ---
        act_load_image = QAction(_(u"Charger &image de calque..."), self)
        act_load_image.setStatusTip(
            _(u"Charger une image (jpg, png, tif...) en arriere-plan pour"
            u" decalquer un profil. Manipulation : touche « i » + souris"))
        act_load_image.triggered.connect(self._on_load_image)
        file_menu.addAction(act_load_image)

        act_clear_image = QAction(_(u"Retirer l'image de calque"), self)
        act_clear_image.setStatusTip(
            _(u"Supprimer l'image de calque actuellement affichee"))
        act_clear_image.triggered.connect(self._on_clear_image)
        file_menu.addAction(act_clear_image)

        file_menu.addSeparator()

        act_quit = QAction(_("&Quitter"), self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.setStatusTip(_(u"Fermer l'application"))
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # --- Edition ---
        edit_menu = menubar.addMenu(_("&Edition"))

        act_undo = QAction(_("&Annuler"), self)
        act_undo.setShortcut("Ctrl+Z")
        act_undo.setStatusTip(_(u"Annuler la derniere action (non implemente)"))
        edit_menu.addAction(act_undo)

        act_redo = QAction(_("&Refaire"), self)
        act_redo.setShortcut("Ctrl+Y")
        act_redo.setStatusTip(
            _(u"Refaire l'action annulee (non implemente)"))
        edit_menu.addAction(act_redo)

        edit_menu.addSeparator()

        act_to_spline = QAction(_("Convertir en Spline"), self)
        act_to_spline.setShortcut("Ctrl+B")
        act_to_spline.setStatusTip(
            _(u"Convertit le profil courant (mode points discrets) en"
            u" splines de Bezier multi-segment editables"))
        act_to_spline.triggered.connect(self._on_convert_to_spline)
        edit_menu.addAction(act_to_spline)

        edit_menu.addSeparator()

        sample_menu = edit_menu.addMenu(_(u"\u00c9chantillonnage"))
        sample_menu.setStatusTip(
            _(u"Modifier le nombre de points d'echantillonnage des splines"))
        act_sample_cur = sample_menu.addAction(_("Profil &courant..."))
        act_sample_cur.setStatusTip(
            _(u"Choisir le nombre de points pour le profil courant"
            u" (necessite mode Spline)"))
        act_sample_cur.triggered.connect(
            lambda: self._on_change_sampling('current'))
        act_sample_ref = sample_menu.addAction(
            _(u"Profil r\u00e9f\u00e9rence..."))
        act_sample_ref.setStatusTip(
            _(u"Choisir le nombre de points pour le profil de reference"
            u" (necessite mode Spline)"))
        act_sample_ref.triggered.connect(
            lambda: self._on_change_sampling('reference'))

        # --- Affichage ---
        view_menu = menubar.addMenu(_("&Affichage"))

        act_zoom_fit = QAction(_("Zoom &adapte"), self)
        act_zoom_fit.setShortcut("Ctrl+0")
        act_zoom_fit.setStatusTip(
            _(u"Recadrer la vue sur l'ensemble des profils visibles"))
        act_zoom_fit.triggered.connect(self._on_zoom_fit)
        view_menu.addAction(act_zoom_fit)

        # Sous-menu Disposition
        disp_menu = view_menu.addMenu(_("&Disposition"))
        disp_menu.setStatusTip(
            _(u"Choisir la grille (lignes x colonnes) de l'onglet Resultats"))
        self._disp_actions = []
        for rows, cols in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]:
            label = "%d \u00d7 %d" % (rows, cols)
            act = QAction(label, self)
            act.setCheckable(True)
            act.setStatusTip(
                _(u"Disposer les analyses en grille %d ligne(s) x %d"
                u" colonne(s)") % (rows, cols))
            if rows == 2 and cols == 2:
                act.setChecked(True)
            act.triggered.connect(
                lambda checked, r=rows, c=cols: self._on_set_grid(r, c))
            disp_menu.addAction(act)
            self._disp_actions.append((rows, cols, act))

        # --- Options ---
        from PySide6.QtGui import QActionGroup
        options_menu = menubar.addMenu(_("&Options"))

        dev_menu = options_menu.addMenu(_(u"Déviation"))
        dev_menu.setStatusTip(
            _(u"Mode de calcul de la deviation entre profil courant et"
            u" reference"))
        self._dev_mode_group = QActionGroup(self)
        self._dev_mode_group.setExclusive(True)

        act_dev_vert = QAction(_(u"Verticale (épaisseur)"), self)
        act_dev_vert.setCheckable(True)
        act_dev_vert.setChecked(True)
        act_dev_vert.setStatusTip(
            _(u"Ecart mesure verticalement (selon z), a abscisse constante"))
        act_dev_vert.triggered.connect(
            lambda: self._on_set_deviation_mode('vertical'))
        self._dev_mode_group.addAction(act_dev_vert)
        dev_menu.addAction(act_dev_vert)

        act_dev_norm = QAction(_(u"Normale (perpendiculaire)"), self)
        act_dev_norm.setCheckable(True)
        act_dev_norm.setStatusTip(
            _(u"Ecart mesure perpendiculairement a la surface du profil"
            u" courant"))
        act_dev_norm.triggered.connect(
            lambda: self._on_set_deviation_mode('normal'))
        self._dev_mode_group.addAction(act_dev_norm)
        dev_menu.addAction(act_dev_norm)

        # Sous-menu Solveur (backend de calcul aerodynamique)
        # N'afficher le choix que si plusieurs backends sont disponibles.
        if len(self._available_solvers) > 1:
            solver_menu = options_menu.addMenu(_(u"&Solveur"))
            solver_menu.setStatusTip(
                _(u"Backend de calcul aerodynamique utilise pour les"
                  u" simulations"))
            self._solver_group = QActionGroup(self)
            self._solver_group.setExclusive(True)
            _solver_labels = {'xfoil': u'XFoil', 'flexfoil': u'FlexFoil'}
            for name in self._available_solvers:
                act_solver = QAction(
                    _solver_labels.get(name, name), self)
                act_solver.setCheckable(True)
                act_solver.setChecked(name == self._solver)
                act_solver.setStatusTip(
                    _(u"Utiliser %s pour les calculs")
                    % _solver_labels.get(name, name))
                act_solver.triggered.connect(
                    lambda checked, n=name: self._on_set_solver(n))
                self._solver_group.addAction(act_solver)
                solver_menu.addAction(act_solver)

        # Sous-menu Langue (effet au redemarrage)
        lang_menu = options_menu.addMenu(_(u"&Langue"))
        lang_menu.setStatusTip(
            _(u"Choisir la langue de l'interface (effet au redemarrage)"))
        self._lang_group = QActionGroup(self)
        self._lang_group.setExclusive(True)
        current_lang = i18n.get_language()
        for code, name in i18n.LANGUAGES:
            act_lang = QAction(name, self)
            act_lang.setCheckable(True)
            act_lang.setChecked(code == current_lang)
            act_lang.triggered.connect(
                lambda checked, c=code: self._on_set_language(c))
            self._lang_group.addAction(act_lang)
            lang_menu.addAction(act_lang)

        # --- Aide ---
        help_menu = menubar.addMenu(_("&Aide"))

        act_manuel = QAction(_(u"&Manuel utilisateur"), self)
        act_manuel.setShortcut("F1")
        act_manuel.setStatusTip(
            _(u"Ouvrir le manuel utilisateur (PDF) avec le visualiseur"
            u" par defaut du systeme"))
        act_manuel.triggered.connect(self._on_open_manuel)
        help_menu.addAction(act_manuel)

        help_menu.addSeparator()

        act_about = QAction(_("A &propos..."), self)
        act_about.setStatusTip(_(u"Informations sur AirfoilTools"))
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
        self._tab_cp = TabCp()

        self._tabs.addTab(self._tab_profils, _("Profils"))
        self._tabs.addTab(self._tab_xfoil, _(u"Param\u00e9trage XFoil"))
        self._tabs.addTab(self._tab_results, _(u"R\u00e9sultats"))
        self._tabs.addTab(self._tab_cp, _(u"Cp / Couche limite"))

        # Connecter le bouton Lancer
        self._tab_xfoil.run_requested.connect(self._on_run_simulations)
        # Connecter les boutons de diagnostic
        self._tab_xfoil.diagnostic_requested.connect(self._on_diagnostic)

        # Marquer les resultats obsoletes quand un profil change
        self._tab_profils.profil_changed.connect(
            self._tab_results.mark_stale)

        # Worker en cours
        self._sim_worker = None
        # Repertoires de travail XFoil par role (pour le diagnostic)
        self._work_dirs = {}

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
            self.statusBar().showMessage(_("Profil %s charge : %s") % (label, info))
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, _("Erreur de chargement"), info)
            self.statusBar().showMessage(_("Echec du chargement"))

    def _on_open_current(self):
        """Ouvre un fichier comme profil courant."""
        self._open_profil("current")

    def _on_open_reference(self):
        """Ouvre un fichier comme profil de reference."""
        self._open_profil("reference")

    def _on_open_uiuc(self, role):
        """Ouvre le dialogue UIUC pour charger un profil depuis le serveur Selig.

        :param role: 'current' ou 'reference'
        """
        from PySide6.QtWidgets import QDialog
        from .dialog_uiuc import DialogUIUC
        label = u"courant" if role == "current" else u"référence"
        dlg = DialogUIUC(parent=self, role_label=label)
        if dlg.exec() != QDialog.Accepted or not dlg.selected_path:
            return
        ok, info = self._tab_profils.load_profil_from_file(
            dlg.selected_path, role)
        if ok:
            self.statusBar().showMessage(
                _(u"Profil %s charge depuis UIUC : %s") % (label, info))
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, _("Erreur de chargement"), info)
            self.statusBar().showMessage(
                _(u"Echec du chargement depuis UIUC"))

    def _on_set_deviation_mode(self, mode):
        u"""Change le mode de calcul de la deviation (vertical / normal).

        :param mode: 'vertical' ou 'normal'
        """
        self._tab_profils.set_deviation_mode(mode)
        label = u"verticale" if mode == 'vertical' else u"normale"
        self.statusBar().showMessage(_(u"Déviation : %s") % label)

    def _on_new_naca(self, role):
        u"""Genere un profil NACA a partir d'indices saisis par l'utilisateur.

        :param role: 'current' ou 'reference'
        """
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        label = u"courant" if role == "current" else u"référence"
        text, ok = QInputDialog.getText(
            self, u"Profil NACA — %s" % label,
            u"Indices NACA (4 ou 5 chiffres, ex. 2412 ou 23012) :")
        if not ok:
            return
        designation = text.strip()
        if not (designation.isdigit() and len(designation) in (4, 5)):
            QMessageBox.warning(
                self, _(u"Indices NACA invalides"),
                _(u"Saisissez 4 ou 5 chiffres (ex. 2412 ou 23012).\n"
                u"Reçu : « %s »") % text)
            return
        n_points, ok = QInputDialog.getInt(
            self, u"Profil NACA — %s" % label,
            u"Nombre de points :", 150, 20, 10000, 10)
        if not ok:
            return
        res_ok, info = self._tab_profils.load_profil_from_naca(
            designation, role, n_points=n_points)
        if res_ok:
            self.statusBar().showMessage(
                _(u"Profil %s généré : %s") % (label, info))
        else:
            QMessageBox.warning(self, _(u"Erreur de génération NACA"), info)
            self.statusBar().showMessage(_(u"Echec de la génération NACA"))

    def _on_save(self):
        """Sauvegarde le profil courant dans un fichier."""
        ok, info = self._tab_profils.save_current_profil()
        if ok is None:
            return  # annule ou pas de profil
        if ok:
            self.statusBar().showMessage(_("Profil sauvegarde : %s") % info)
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, _("Erreur de sauvegarde"), info)
            self.statusBar().showMessage(_("Echec de la sauvegarde"))

    def _on_save_flap(self):
        """Sauvegarde le profil avec volet (comme le profil courant)."""
        ok, info = self._tab_profils.save_flap_profil()
        if ok is None:
            if info:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, _("Profil avec volet"), info)
            return
        if ok:
            self.statusBar().showMessage(
                _("Profil avec volet sauvegarde : %s") % info)
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, _("Erreur de sauvegarde"), info)
            self.statusBar().showMessage(_("Echec de la sauvegarde"))

    def _on_open_project(self):
        """Ouvre un projet AirfoilTools (.aftproj)."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir un projet",
            "",
            u"Projet AirfoilTools (*.aftproj);;Tous (*)")
        if not filepath:
            return
        ok, info = self._tab_profils.open_project(filepath)
        if ok:
            self._tabs.setCurrentWidget(self._tab_profils)
            self.statusBar().showMessage(_(u"Projet ouvert : %s") % info)
        else:
            QMessageBox.warning(self, _("Erreur d'ouverture du projet"), info)
            self.statusBar().showMessage(_(u"Echec de l'ouverture du projet"))

    def _on_save_project(self):
        """Enregistre le projet courant (profils + image)."""
        from PySide6.QtWidgets import QMessageBox
        ok, info = self._tab_profils.save_project()
        if ok is None:
            return
        if ok:
            self.statusBar().showMessage(_(u"Projet enregistre : %s") % info)
        else:
            QMessageBox.warning(self, _("Erreur d'enregistrement"), info)
            self.statusBar().showMessage(_(u"Echec de l'enregistrement"))

    def _on_load_image(self):
        """Charge une image de calque en arriere-plan."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Charger une image de calque",
            "",
            u"Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;Tous (*)")
        if not filepath:
            return
        ok, info = self._tab_profils.load_background_image(filepath)
        if ok:
            self.statusBar().showMessage(
                _(u"Image de calque chargee : %s "
                u"(maintenir « i » + souris pour ajuster)") % info)
        else:
            QMessageBox.warning(
                self, _("Erreur de chargement de l'image"), info)
            self.statusBar().showMessage(_(u"Echec du chargement de l'image"))

    def _on_clear_image(self):
        """Retire l'image de calque."""
        self._tab_profils.clear_background_image()
        self.statusBar().showMessage(_(u"Image de calque retiree"))

    def _on_change_sampling(self, role):
        """Change le nombre de points d'echantillonnage d'un profil Bezier."""
        ok, info = self._tab_profils.change_sampling(role)
        if ok is None:
            self.statusBar().showMessage(info or "")
        elif ok:
            self.statusBar().showMessage(
                _(u"\u00c9chantillonnage : %s") % info)
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, _(u"\u00c9chantillonnage"), info)

    def _on_convert_to_spline(self):
        """Convertit le profil courant en mode Spline."""
        p = self._tab_profils.profil_current
        if p is None or p.has_splines:
            self.statusBar().showMessage(
                _(u"Pas de profil ou d\u00e9j\u00e0 en mode Spline"))
            return

        from PySide6.QtWidgets import (
            QDialog, QDialogButtonBox, QFormLayout, QSpinBox,
            QDoubleSpinBox
        )
        dlg = QDialog(self)
        dlg.setWindowTitle(_("Convertir en Spline"))
        form = QFormLayout(dlg)

        spn_ext = QSpinBox()
        spn_ext.setRange(2, 30)
        spn_ext.setValue(11)
        spn_ext.setToolTip(
            _(u"Degre des courbes de Bezier de l'extrados.\n"
            u"  Petit (3-5) : forme tres lissee, peu d'inflexions\n"
            u"  Moyen (6-12) : compromis precision/lissage (defaut)\n"
            u"  Grand (>15) : forme tres precise, risque d'oscillations"))
        form.addRow(_(u"Degr\u00e9 extrados :"), spn_ext)

        spn_int = QSpinBox()
        spn_int.setRange(2, 30)
        spn_int.setValue(11)
        spn_int.setToolTip(
            _(u"Degre des courbes de Bezier de l'intrados.\n"
            u"Voir l'aide du degre extrados."))
        form.addRow(_(u"Degr\u00e9 intrados :"), spn_int)

        spn_tol = QDoubleSpinBox()
        spn_tol.setRange(0.0001, 0.1)
        spn_tol.setDecimals(4)
        spn_tol.setSingleStep(0.0005)
        spn_tol.setValue(0.001)
        spn_tol.setToolTip(
            _(u"D\u00e9viation max acceptable (corde = 1).\n"
            u"Plus petit = plus de segments."))
        form.addRow(_(u"Tol\u00e9rance :"), spn_tol)

        spn_max_seg = QSpinBox()
        spn_max_seg.setRange(1, 20)
        spn_max_seg.setValue(1)
        spn_max_seg.setToolTip(
            _(u"Nombre max de segments B\u00e9zier par c\u00f4t\u00e9.\n"
            u"1 = un seul segment (ancien comportement)."))
        form.addRow(_("Max segments :"), spn_max_seg)

        spn_smooth = QDoubleSpinBox()
        spn_smooth.setRange(0.0, 1.0)
        spn_smooth.setDecimals(2)
        spn_smooth.setSingleStep(0.01)
        spn_smooth.setValue(0.1)
        spn_smooth.setToolTip(
            _(u"R\u00e9gularisation (0 = pas de lissage).\n"
            u"Valeurs > 0 lissent le polygone de contr\u00f4le."))
        form.addRow(_("Lissage :"), spn_smooth)

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
                _(u"Pas de profil ou d\u00e9j\u00e0 en mode Spline"))
        elif ok:
            self.statusBar().showMessage(
                _("Profil '%s' converti en Spline") % info)
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, _("Erreur de conversion"), info)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _on_run_simulations(self, target='both'):
        u"""Lance les simulations XFoil pour les profils demandes.

        :param target: 'both', 'current' ou 'reference'
        :type target: str
        """
        if self._sim_worker is not None and self._sim_worker.isRunning():
            self.statusBar().showMessage(_("Simulation deja en cours..."))
            return

        profils = {}
        no_normalize = set()
        if target in ('both', 'current'):
            profils['current'] = self._tab_profils.profil_current
        if target in ('both', 'reference'):
            profils['reference'] = self._tab_profils.profil_reference
        if target in ('both', 'flap'):
            # Volet construit dans le repere normalise du courant : XFoil
            # ne doit ni renormaliser ni redresser le braquage.
            profils['flap'] = self._tab_profils.profil_flap_normalized()
            no_normalize.add('flap')

        # Retirer les None (ex. volet inactif)
        profils = {k: v for k, v in profils.items() if v is not None}
        if not profils:
            self.statusBar().showMessage(_("Aucun profil a simuler"))
            return

        self._sim_target = target
        params = self._tab_xfoil.get_params()

        from .simulation_worker import SimulationWorker
        self._sim_worker = SimulationWorker(
            profils, params, parent=self, no_normalize_roles=no_normalize,
            solver=self._solver)
        self._sim_worker.progress.connect(self._on_sim_progress)
        self._sim_worker.finished_ok.connect(self._on_sim_finished)
        self._sim_worker.finished_error.connect(self._on_sim_error)

        self._tab_xfoil.set_enabled(False)
        self.statusBar().showMessage(_("Simulations en cours..."))
        self._sim_worker.start()

    def _on_sim_progress(self, msg):
        """Met a jour la barre de statut pendant la simulation."""
        self.statusBar().showMessage(msg)

    def _on_sim_finished(self, results):
        u"""Traite les resultats des simulations."""
        self._tab_xfoil.set_enabled(True)
        if self._sim_worker is not None:
            self._work_dirs.update(self._sim_worker.work_dirs)
        self._tab_xfoil.set_diagnostic_available(self._work_dirs.keys())
        self._sim_worker = None

        n_profils = len(results)
        total_pts = sum(r.n_converged for r in results.values())
        self.statusBar().showMessage(
            _("Terminees : %d profil(s), %d pts converges")
            % (n_profils, total_pts))

        # Merge partiel ou remplacement complet
        target = getattr(self, '_sim_target', 'both')
        if target == 'both':
            self._tab_results.set_results(results)
            self._tab_cp.set_results(results)
        else:
            self._tab_results.update_results(results)
            self._tab_cp.update_results(results)

        # Basculer sur l'onglet Resultats
        self._tabs.setCurrentWidget(self._tab_results)

    def _on_sim_error(self, error_msg):
        """Affiche l'erreur de simulation."""
        self._tab_xfoil.set_enabled(True)
        # Conserver les repertoires deja crees : le log d'un calcul en
        # echec est justement ce qu'on veut diagnostiquer.
        if self._sim_worker is not None:
            self._work_dirs.update(self._sim_worker.work_dirs)
        self._tab_xfoil.set_diagnostic_available(self._work_dirs.keys())
        self._sim_worker = None

        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(
            self, _("Erreur de simulation"),
            _(u"%s\n\nUtilisez les boutons « Diagnostic » de l'onglet"
            u" Paramétrage XFoil pour consulter le log.") % error_msg)
        self.statusBar().showMessage(_("Echec de la simulation"))

    def _on_diagnostic(self, role):
        u"""Ouvre le dialogue de diagnostic pour un profil.

        :param role: 'current', 'reference' ou 'flap'
        :type role: str
        """
        work_dir = self._work_dirs.get(role)
        labels = {'current': 'courant', 'reference': u'référence',
                  'flap': 'volet'}
        label = labels.get(role, role)
        if not work_dir:
            self.statusBar().showMessage(
                _(u"Aucune simulation pour le profil %s") % label)
            return
        from .diagnostic_dialog import DiagnosticDialog
        dlg = DiagnosticDialog(work_dir, title=label, parent=self)
        dlg.exec()

    def _on_set_grid(self, rows, cols):
        """Change la disposition de la grille de resultats."""
        self._tab_results.set_grid(rows, cols)
        # Mettre a jour les coches du menu
        for r, c, act in self._disp_actions:
            act.setChecked(r == rows and c == cols)

    def _on_zoom_fit(self):
        """Zoom adapte sur le canvas."""
        self._tab_profils.zoom_fit()

    def _on_set_language(self, code):
        """Enregistre la langue choisie (prise en compte au redemarrage)."""
        from PySide6.QtWidgets import QMessageBox
        if code == i18n.get_language():
            return
        i18n.save_language(code)
        QMessageBox.information(
            self, _(u"Langue de l'interface"),
            _(u"La langue sera appliquee au prochain demarrage"
              u" d'AirfoilTools."))

    def _on_set_solver(self, name):
        u"""Change le backend de calcul et le persiste (effet immediat)."""
        from PySide6.QtCore import QSettings
        self._solver = name
        QSettings().setValue('solver', name)
        labels = {'xfoil': u'XFoil', 'flexfoil': u'FlexFoil'}
        self.statusBar().showMessage(
            _(u"Solveur : %s") % labels.get(name, name))

    def _on_open_manuel(self):
        """Ouvre le manuel utilisateur PDF avec le visualiseur par defaut."""
        path = _find_manuel_pdf(i18n.get_language())
        if path is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, _(u"Manuel introuvable"),
                _(u"Le fichier manuel.pdf n'a pas ete trouve.\n\n"
                u"Verifiez que le fichier docs/manuel/manuel.pdf "
                u"existe a cote de l'application."))
            return
        try:
            _open_with_default_app(path)
            self.statusBar().showMessage(
                _(u"Manuel ouvert : %s") % path)
        except OSError as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, _(u"Impossible d'ouvrir le manuel"),
                _(u"Erreur lors de l'ouverture du PDF :\n\n%s") % exc)

    def _on_about(self):
        """Affiche la boite A propos."""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self, _("AirfoilTools"),
            _(u"AirfoilTools - Analyse a\u00e9rodynamique 2D\n"
              u"Courbes de B\u00e9zier, profils, XFoil\n\n")
            + (u"Version %s\n" % __version__)
            + _(u"Premi\u00e8re version : 2022\n\n")
            + _(u"Auteur : Beno\u00eet Gagnaire")
        )


def main():
    """Point d'entree de l'application."""
    app = QApplication.instance() or QApplication(sys.argv)
    # Identite pour QSettings (registre Windows) + langue persistee.
    app.setOrganizationName("Nervures")
    app.setApplicationName("AirfoilTools")
    i18n.load_language()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
