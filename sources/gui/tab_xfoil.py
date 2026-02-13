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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        u"""Construit l'interface de l'onglet."""
        layout = QVBoxLayout(self)

        # --- Reynolds ---
        grp_re = QGroupBox("Reynolds")
        grid_re = QGridLayout(grp_re)

        grid_re.addWidget(QLabel("Min :"), 0, 0)
        self._spn_re_min = QSpinBox()
        self._spn_re_min.setRange(1000, 100000000)
        self._spn_re_min.setSingleStep(10000)
        self._spn_re_min.setValue(100000)
        grid_re.addWidget(self._spn_re_min, 0, 1)

        grid_re.addWidget(QLabel("Max :"), 0, 2)
        self._spn_re_max = QSpinBox()
        self._spn_re_max.setRange(1000, 100000000)
        self._spn_re_max.setSingleStep(10000)
        self._spn_re_max.setValue(200000)
        grid_re.addWidget(self._spn_re_max, 0, 3)

        grid_re.addWidget(QLabel("Pas :"), 0, 4)
        self._spn_re_step = QSpinBox()
        self._spn_re_step.setRange(1000, 100000000)
        self._spn_re_step.setSingleStep(10000)
        self._spn_re_step.setValue(100000)
        grid_re.addWidget(self._spn_re_step, 0, 5)

        layout.addWidget(grp_re)

        # --- Alpha ---
        grp_alpha = QGroupBox("Alpha (deg)")
        grid_alpha = QGridLayout(grp_alpha)

        grid_alpha.addWidget(QLabel("Min :"), 0, 0)
        self._spn_alpha_min = QDoubleSpinBox()
        self._spn_alpha_min.setRange(-30.0, 30.0)
        self._spn_alpha_min.setSingleStep(0.5)
        self._spn_alpha_min.setDecimals(1)
        self._spn_alpha_min.setValue(0.0)
        grid_alpha.addWidget(self._spn_alpha_min, 0, 1)

        grid_alpha.addWidget(QLabel("Max :"), 0, 2)
        self._spn_alpha_max = QDoubleSpinBox()
        self._spn_alpha_max.setRange(-30.0, 30.0)
        self._spn_alpha_max.setSingleStep(0.5)
        self._spn_alpha_max.setDecimals(1)
        self._spn_alpha_max.setValue(5.0)
        grid_alpha.addWidget(self._spn_alpha_max, 0, 3)

        grid_alpha.addWidget(QLabel("Pas :"), 0, 4)
        self._spn_alpha_step = QDoubleSpinBox()
        self._spn_alpha_step.setRange(0.1, 2.0)
        self._spn_alpha_step.setSingleStep(0.1)
        self._spn_alpha_step.setDecimals(1)
        self._spn_alpha_step.setValue(0.5)
        grid_alpha.addWidget(self._spn_alpha_step, 0, 5)

        layout.addWidget(grp_alpha)

        # --- Parametres XFoil ---
        grp_xfoil = QGroupBox(u"Param\u00e8tres XFoil")
        grid_xf = QGridLayout(grp_xfoil)

        # Viscosite
        grid_xf.addWidget(QLabel(u"Viscosit\u00e9 :"), 0, 0)
        self._chk_viscous = QCheckBox("Avec")
        self._chk_viscous.setChecked(True)
        grid_xf.addWidget(self._chk_viscous, 0, 1)

        # NCRIT
        grid_xf.addWidget(QLabel("NCRIT :"), 0, 2)
        self._spn_ncrit = QDoubleSpinBox()
        self._spn_ncrit.setRange(0.1, 20.0)
        self._spn_ncrit.setSingleStep(1.0)
        self._spn_ncrit.setDecimals(1)
        self._spn_ncrit.setValue(9.0)
        grid_xf.addWidget(self._spn_ncrit, 0, 3)

        # XTR_TOP
        grid_xf.addWidget(QLabel("XTR Top :"), 1, 0)
        self._spn_xtr_top = QDoubleSpinBox()
        self._spn_xtr_top.setRange(0.0, 1.0)
        self._spn_xtr_top.setSingleStep(0.01)
        self._spn_xtr_top.setDecimals(3)
        self._spn_xtr_top.setValue(0.01)
        grid_xf.addWidget(self._spn_xtr_top, 1, 1)

        # XTR_BOT
        grid_xf.addWidget(QLabel("XTR Bot :"), 1, 2)
        self._spn_xtr_bot = QDoubleSpinBox()
        self._spn_xtr_bot.setRange(0.0, 1.0)
        self._spn_xtr_bot.setSingleStep(0.01)
        self._spn_xtr_bot.setDecimals(3)
        self._spn_xtr_bot.setValue(0.01)
        grid_xf.addWidget(self._spn_xtr_bot, 1, 3)

        # REPANEL
        grid_xf.addWidget(QLabel("Repanel :"), 2, 0)
        self._chk_repanel = QCheckBox("Oui")
        self._chk_repanel.setChecked(True)
        grid_xf.addWidget(self._chk_repanel, 2, 1)

        # NPANEL
        grid_xf.addWidget(QLabel("NPanel :"), 2, 2)
        self._spn_npanel = QSpinBox()
        self._spn_npanel.setRange(50, 500)
        self._spn_npanel.setSingleStep(10)
        self._spn_npanel.setValue(200)
        grid_xf.addWidget(self._spn_npanel, 2, 3)

        # TIMEOUT
        grid_xf.addWidget(QLabel("Timeout (s) :"), 3, 0)
        self._spn_timeout = QSpinBox()
        self._spn_timeout.setRange(5, 600)
        self._spn_timeout.setSingleStep(10)
        self._spn_timeout.setValue(30)
        grid_xf.addWidget(self._spn_timeout, 3, 1)

        layout.addWidget(grp_xfoil)

        # --- Boutons Lancer ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self._btn_run_both = QPushButton("Lancer les 2")
        self._btn_run_both.clicked.connect(
            lambda: self.run_requested.emit('both'))
        btn_layout.addWidget(self._btn_run_both)

        self._btn_run_current = QPushButton("Courant seul")
        self._btn_run_current.clicked.connect(
            lambda: self.run_requested.emit('current'))
        btn_layout.addWidget(self._btn_run_current)

        self._btn_run_reference = QPushButton(u"R\u00e9f\u00e9rence seul")
        self._btn_run_reference.clicked.connect(
            lambda: self.run_requested.emit('reference'))
        btn_layout.addWidget(self._btn_run_reference)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

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
        }

    def set_enabled(self, enabled):
        u"""Active/desactive les controles (pendant une simulation)."""
        for w in self.findChildren(QWidget):
            w.setEnabled(enabled)
