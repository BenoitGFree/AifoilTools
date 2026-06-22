#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Onglet Parametrage XFoil : saisie des parametres de simulation."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton
)
from PySide6.QtCore import Qt, Signal


class TabXfoil(QWidget):
    u"""Onglet de parametrage des simulations XFoil.

    Expose les parametres via get_params() sous forme de dict
    compatible avec XFoilPreprocessor.prepare().
    """

    # Signal emis quand l'utilisateur clique sur Lancer
    # target: 'both', 'current' ou 'reference'
    run_requested = Signal(str)

    # Signal emis pour consulter le diagnostic d'un profil
    # role: 'current', 'reference' ou 'flap'
    diagnostic_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        u"""Construit l'interface de l'onglet."""
        layout = QVBoxLayout(self)

        # --- Reynolds ---
        grp_re = QGroupBox("Reynolds")
        grid_re = QGridLayout(grp_re)

        grp_re.setToolTip(
            u"Plage de nombres de Reynolds pour le balayage.\n"
            u"Une polaire est calculee pour chaque Reynolds entre Min"
            u" et Max, par increments de Pas.")

        grid_re.addWidget(QLabel("Min :"), 0, 0)
        self._spn_re_min = QSpinBox()
        self._spn_re_min.setRange(1000, 100000000)
        self._spn_re_min.setSingleStep(10000)
        self._spn_re_min.setValue(100000)
        self._spn_re_min.setToolTip(u"Reynolds minimum du balayage.")
        grid_re.addWidget(self._spn_re_min, 0, 1)

        grid_re.addWidget(QLabel("Max :"), 0, 2)
        self._spn_re_max = QSpinBox()
        self._spn_re_max.setRange(1000, 100000000)
        self._spn_re_max.setSingleStep(10000)
        self._spn_re_max.setValue(200000)
        self._spn_re_max.setToolTip(u"Reynolds maximum du balayage.")
        grid_re.addWidget(self._spn_re_max, 0, 3)

        grid_re.addWidget(QLabel("Pas :"), 0, 4)
        self._spn_re_step = QSpinBox()
        self._spn_re_step.setRange(1000, 100000000)
        self._spn_re_step.setSingleStep(10000)
        self._spn_re_step.setValue(100000)
        self._spn_re_step.setToolTip(
            u"Increment entre deux Reynolds successifs.\n"
            u"Mettre Pas >= (Max - Min) pour ne calculer qu'une seule"
            u" valeur.")
        grid_re.addWidget(self._spn_re_step, 0, 5)

        layout.addWidget(grp_re)

        # --- Alpha ---
        grp_alpha = QGroupBox("Alpha (deg)")
        grp_alpha.setToolTip(
            u"Plage d'angles d'incidence (en degres) pour le balayage"
            u" de chaque polaire.")
        grid_alpha = QGridLayout(grp_alpha)

        grid_alpha.addWidget(QLabel("Min :"), 0, 0)
        self._spn_alpha_min = QDoubleSpinBox()
        self._spn_alpha_min.setRange(-30.0, 30.0)
        self._spn_alpha_min.setSingleStep(0.5)
        self._spn_alpha_min.setDecimals(1)
        self._spn_alpha_min.setValue(0.0)
        self._spn_alpha_min.setToolTip(u"Angle d'incidence minimum (deg).")
        grid_alpha.addWidget(self._spn_alpha_min, 0, 1)

        grid_alpha.addWidget(QLabel("Max :"), 0, 2)
        self._spn_alpha_max = QDoubleSpinBox()
        self._spn_alpha_max.setRange(-30.0, 30.0)
        self._spn_alpha_max.setSingleStep(0.5)
        self._spn_alpha_max.setDecimals(1)
        self._spn_alpha_max.setValue(5.0)
        self._spn_alpha_max.setToolTip(u"Angle d'incidence maximum (deg).")
        grid_alpha.addWidget(self._spn_alpha_max, 0, 3)

        grid_alpha.addWidget(QLabel("Pas :"), 0, 4)
        self._spn_alpha_step = QDoubleSpinBox()
        self._spn_alpha_step.setRange(0.1, 2.0)
        self._spn_alpha_step.setSingleStep(0.1)
        self._spn_alpha_step.setDecimals(1)
        self._spn_alpha_step.setValue(0.5)
        self._spn_alpha_step.setToolTip(
            u"Increment d'angle entre deux points de la polaire.\n"
            u"Plus petit = polaire plus fine mais simulation plus longue.")
        grid_alpha.addWidget(self._spn_alpha_step, 0, 5)

        self._chk_aseq = QCheckBox("ASEQ")
        self._chk_aseq.setChecked(True)
        self._chk_aseq.setToolTip(
            u"Coche : balayage ASEQ de XFoil (rapide, comportement"
            u" historique).\n"
            u"Decoche : ALFA individuel depuis alpha=0 puis branches"
            u" positive et negative separees (plus robuste a la"
            u" convergence aux angles eleves).")
        grid_alpha.addWidget(self._chk_aseq, 0, 6)

        self._chk_init_retry = QCheckBox(u"Réinit. si échec")
        self._chk_init_retry.setChecked(False)
        self._chk_init_retry.setToolTip(
            u"Coche : apres le balayage en marche, une 2e passe rejoue"
            u" chaque incidence depuis un etat propre (INIT avant chaque"
            u" ALFA).\n"
            u"Recupere les points qu'une non-convergence en marche"
            u" ('VISCAL: Convergence failed') aurait fait sauter, sans"
            u" degrader le balayage en marche (on garde sa solution).\n"
            u"Implique le mode ALFA individuel (incompatible avec ASEQ)"
            u" et environ double le temps de calcul.")
        grid_alpha.addWidget(self._chk_init_retry, 0, 7)

        layout.addWidget(grp_alpha)

        # --- Parametres XFoil ---
        grp_xfoil = QGroupBox(u"Param\u00e8tres XFoil")
        grp_xfoil.setToolTip(
            u"Parametres avances passes a XFoil.\n"
            u"Ne modifier que si vous savez ce que vous faites.")
        grid_xf = QGridLayout(grp_xfoil)

        # Viscosite
        grid_xf.addWidget(QLabel(u"Viscosit\u00e9 :"), 0, 0)
        self._chk_viscous = QCheckBox("Avec")
        self._chk_viscous.setChecked(True)
        self._chk_viscous.setToolTip(
            u"Coche : analyse visqueuse (commande VISC, prend en compte"
            u" la couche limite).\n"
            u"Decoche : analyse non-visqueuse (potentielle, sans"
            u" frottement). Toujours utiliser visqueux pour des polaires"
            u" realistes.")
        grid_xf.addWidget(self._chk_viscous, 0, 1)

        # NCRIT
        grid_xf.addWidget(QLabel("NCRIT :"), 0, 2)
        self._spn_ncrit = QDoubleSpinBox()
        self._spn_ncrit.setRange(0.1, 20.0)
        self._spn_ncrit.setSingleStep(1.0)
        self._spn_ncrit.setDecimals(1)
        self._spn_ncrit.setValue(9.0)
        self._spn_ncrit.setToolTip(
            u"Critere de transition de couche limite (methode e^N).\n"
            u"  9 : ecoulement libre standard (defaut)\n"
            u"  4-7 : turbulence ambiante elevee (souffleries)\n"
            u" 12-14 : ecoulement tres calme")
        grid_xf.addWidget(self._spn_ncrit, 0, 3)

        # XTR_TOP
        grid_xf.addWidget(QLabel("XTR Top :"), 1, 0)
        self._spn_xtr_top = QDoubleSpinBox()
        self._spn_xtr_top.setRange(0.05, 1.0)
        self._spn_xtr_top.setSingleStep(0.05)
        self._spn_xtr_top.setDecimals(3)
        self._spn_xtr_top.setValue(0.05)
        self._spn_xtr_top.setToolTip(
            u"Position de transition forcee sur l'extrados,"
            u" en fraction de corde (x/c).\n"
            u"1.0 = transition libre (laissee au critere NCRIT).\n"
            u"Valeur < 1.0 = trip turbulent (rugosite/turbulateur).\n"
            u"Min 0.05 : forcer la transition plus pres du bord"
            u" d'attaque destabilise XFoil (NaN, boucle infinie).")
        grid_xf.addWidget(self._spn_xtr_top, 1, 1)

        # XTR_BOT
        grid_xf.addWidget(QLabel("XTR Bot :"), 1, 2)
        self._spn_xtr_bot = QDoubleSpinBox()
        self._spn_xtr_bot.setRange(0.05, 1.0)
        self._spn_xtr_bot.setSingleStep(0.05)
        self._spn_xtr_bot.setDecimals(3)
        self._spn_xtr_bot.setValue(0.05)
        self._spn_xtr_bot.setToolTip(
            u"Position de transition forcee sur l'intrados,"
            u" en fraction de corde (x/c).\n"
            u"1.0 = transition libre. Valeur < 1.0 = trip turbulent.\n"
            u"Min 0.05 (stabilite XFoil).")
        grid_xf.addWidget(self._spn_xtr_bot, 1, 3)

        # REPANEL
        grid_xf.addWidget(QLabel("Repanel :"), 2, 0)
        self._chk_repanel = QCheckBox("Oui")
        self._chk_repanel.setChecked(True)
        self._chk_repanel.setToolTip(
            u"Coche : XFoil reechantillonne le profil (commande PANE)"
            u" avant simulation.\n"
            u"Recommande pour les profils a echantillonnage non-uniforme.")
        grid_xf.addWidget(self._chk_repanel, 2, 1)

        # NPANEL
        grid_xf.addWidget(QLabel("NPanel :"), 2, 2)
        self._spn_npanel = QSpinBox()
        self._spn_npanel.setRange(50, 500)
        self._spn_npanel.setSingleStep(10)
        self._spn_npanel.setValue(200)
        self._spn_npanel.setToolTip(
            u"Nombre de panneaux apres reechantillonnage (PANE).\n"
            u"Defaut : 200. Plus = plus precis mais plus lent.\n"
            u"Sans effet si Repanel decoche.")
        grid_xf.addWidget(self._spn_npanel, 2, 3)

        # TIMEOUT
        grid_xf.addWidget(QLabel("Timeout (s) :"), 3, 0)
        self._spn_timeout = QSpinBox()
        self._spn_timeout.setRange(5, 600)
        self._spn_timeout.setSingleStep(10)
        self._spn_timeout.setValue(30)
        self._spn_timeout.setToolTip(
            u"Duree maximale d'execution de XFoil pour une polaire,"
            u" en secondes.\n"
            u"Au-dela, le processus est tue (utile en cas de"
            u" non-convergence).")
        grid_xf.addWidget(self._spn_timeout, 3, 1)

        # ITER
        grid_xf.addWidget(QLabel(u"Iterations :"), 3, 2)
        self._spn_iter = QSpinBox()
        self._spn_iter.setRange(10, 2000)
        self._spn_iter.setSingleStep(10)
        self._spn_iter.setValue(100)
        self._spn_iter.setToolTip(
            u"Nombre maximum d'iterations de Newton par point (commande"
            u" ITER de XFoil).\n"
            u"Defaut : 100. Augmenter (200-300) si un point precis ne"
            u" converge pas (message 'VISCAL: Convergence failed'),\n"
            u"typique pres du decrochage ou avec un coude de volet.")
        grid_xf.addWidget(self._spn_iter, 3, 3)

        layout.addWidget(grp_xfoil)

        # --- Boutons Lancer ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._btn_run_both = QPushButton("Lancer tout")
        self._btn_run_both.setToolTip(
            u"Lance la simulation XFoil sur tous les profils actifs :"
            u" courant, reference et, si le volet est active, le profil"
            u" avec volet.\n"
            u"Les resultats apparaitront dans l'onglet Resultats.")
        self._btn_run_both.clicked.connect(
            lambda: self.run_requested.emit('both'))
        btn_layout.addWidget(self._btn_run_both)

        self._btn_run_current = QPushButton("Courant seul")
        self._btn_run_current.setToolTip(
            u"Lance la simulation uniquement sur le profil courant"
            u" (bleu).")
        self._btn_run_current.clicked.connect(
            lambda: self.run_requested.emit('current'))
        btn_layout.addWidget(self._btn_run_current)

        self._btn_run_reference = QPushButton(u"R\u00e9f\u00e9rence seul")
        self._btn_run_reference.setToolTip(
            u"Lance la simulation uniquement sur le profil de"
            u" reference (rouge).")
        self._btn_run_reference.clicked.connect(
            lambda: self.run_requested.emit('reference'))
        btn_layout.addWidget(self._btn_run_reference)

        self._btn_run_flap = QPushButton("Flap seul")
        self._btn_run_flap.setToolTip(
            u"Lance la simulation uniquement sur le profil avec volet"
            u" (vert).\nNecessite que le volet soit active dans l'onglet"
            u" Profils.")
        self._btn_run_flap.clicked.connect(
            lambda: self.run_requested.emit('flap'))
        btn_layout.addWidget(self._btn_run_flap)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # --- Diagnostic ---
        grp_diag = QGroupBox("Diagnostic")
        grp_diag.setToolTip(
            u"Acces au repertoire de travail et au log console XFoil de"
            u" la derniere simulation de chaque profil.\n"
            u"Utile pour comprendre une non-convergence ou une erreur.")
        diag_layout = QHBoxLayout(grp_diag)
        diag_layout.addWidget(QLabel(u"Log / fichiers :"))
        diag_layout.addStretch()

        self._btn_diag_current = QPushButton("Courant")
        self._btn_diag_current.setToolTip(
            u"Ouvre le log et les fichiers XFoil du profil courant.")
        self._btn_diag_current.clicked.connect(
            lambda: self.diagnostic_requested.emit('current'))
        diag_layout.addWidget(self._btn_diag_current)

        self._btn_diag_reference = QPushButton(u"Référence")
        self._btn_diag_reference.setToolTip(
            u"Ouvre le log et les fichiers XFoil du profil de reference.")
        self._btn_diag_reference.clicked.connect(
            lambda: self.diagnostic_requested.emit('reference'))
        diag_layout.addWidget(self._btn_diag_reference)

        self._btn_diag_flap = QPushButton("Volet")
        self._btn_diag_flap.setToolTip(
            u"Ouvre le log et les fichiers XFoil du profil avec volet.")
        self._btn_diag_flap.clicked.connect(
            lambda: self.diagnostic_requested.emit('flap'))
        diag_layout.addWidget(self._btn_diag_flap)

        self._diag_buttons = {
            'current': self._btn_diag_current,
            'reference': self._btn_diag_reference,
            'flap': self._btn_diag_flap,
        }
        # Desactives tant qu'aucune simulation n'a tourne
        for b in self._diag_buttons.values():
            b.setEnabled(False)

        layout.addWidget(grp_diag)

        layout.addStretch()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def get_params(self):
        u"""Retourne les parametres sous forme de dict.

        Compatible avec XFoilPreprocessor.prepare(params=...).

        :returns: dict des parametres
        :rtype: dict
        """
        # Construire la liste des Reynolds
        re_min = self._spn_re_min.value()
        re_max = self._spn_re_max.value()
        re_step = self._spn_re_step.value()

        re_list = []
        re_val = re_min
        while re_val <= re_max + 0.5:
            re_list.append(float(re_val))
            re_val += re_step
        if not re_list:
            re_list = [float(re_min)]

        return {
            'RE_LIST': re_list,
            'RE': re_list[0],
            'ALPHA_MIN': self._spn_alpha_min.value(),
            'ALPHA_MAX': self._spn_alpha_max.value(),
            'ALPHA_STEP': self._spn_alpha_step.value(),
            'VISCOUS': self._chk_viscous.isChecked(),
            'NCRIT': self._spn_ncrit.value(),
            'XTR_TOP': self._spn_xtr_top.value(),
            'XTR_BOT': self._spn_xtr_bot.value(),
            'REPANEL': self._chk_repanel.isChecked(),
            'NPANEL': self._spn_npanel.value(),
            'TIMEOUT': self._spn_timeout.value(),
            'ITER': self._spn_iter.value(),
            'USE_ASEQ': self._chk_aseq.isChecked(),
            'INIT_RETRY': self._chk_init_retry.isChecked(),
        }

    def set_enabled(self, enabled):
        u"""Active/desactive les controles (pendant une simulation)."""
        for w in self.findChildren(QWidget):
            w.setEnabled(enabled)

    def set_diagnostic_available(self, roles):
        u"""Active les boutons de diagnostic des roles disponibles.

        :param roles: iterable des roles ayant un repertoire de travail
                      ('current', 'reference', 'flap')
        :type roles: iterable
        """
        available = set(roles)
        for role, btn in self._diag_buttons.items():
            btn.setEnabled(role in available)
