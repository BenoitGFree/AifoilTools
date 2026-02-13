#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Worker thread pour lancer les simulations XFoil sans bloquer la GUI."""

import logging

from PySide6.QtCore import QThread, Signal

from model.simulation import Simulation, SimulationResults

logger = logging.getLogger(__name__)


class SimulationWorker(QThread):
    u"""Execute les simulations pour profil courant et reference en arriere-plan.

    Signaux :
    - progress(str) : message de progression
    - finished_ok(dict) : resultats {role: SimulationResults}
    - finished_error(str) : message d'erreur
    """

    progress = Signal(str)
    finished_ok = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, profils, params, parent=None):
        u"""
        :param profils: dict {'current': Profil, 'reference': Profil}
                        (les valeurs peuvent etre None)
        :param params: dict de parametres XFoil
        """
        super().__init__(parent)
        self._profils = profils
        self._params = params

    def run(self):
        u"""Execute les simulations (dans le thread)."""
        results = {}
        try:
            for role, profil in self._profils.items():
                if profil is None:
                    continue
                label = profil.name or role
                self.progress.emit("Simulation '%s'..." % label)
                logger.info("Lancement simulation %s (%s)", role, label)

                sim = Simulation(profil, params=self._params)
                sim_results = sim.run()
                results[role] = sim_results

                n_pts = sim_results.n_converged
                self.progress.emit(
                    "'%s' : %d points converges" % (label, n_pts))

            self.finished_ok.emit(results)

        except Exception as e:
            logger.error("Erreur simulation : %s", str(e))
            self.finished_error.emit(str(e))
