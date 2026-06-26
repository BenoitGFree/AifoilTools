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
    - progress_step(int, int) : (profils termines, total) pour la barre
    - finished_ok(dict) : resultats {role: SimulationResults}
    - finished_error(str) : message d'erreur
    """

    progress = Signal(str)
    progress_step = Signal(int, int)
    finished_ok = Signal(dict)
    finished_error = Signal(str)

    def __init__(self, profils, params, parent=None, no_normalize_roles=None,
                 solver='xfoil'):
        u"""
        :param profils: dict {'current': Profil, 'reference': Profil}
                        (les valeurs peuvent etre None)
        :param params: dict de parametres XFoil
        :param no_normalize_roles: ensemble de roles a NE PAS normaliser
                        avant XFoil (ex. {'flap'} pour conserver le
                        braquage dans le repere du profil courant)
        :param solver: nom du backend de calcul ('xfoil' ou 'flexfoil')
        """
        super().__init__(parent)
        self._profils = profils
        self._params = params
        self._no_normalize_roles = set(no_normalize_roles or ())
        self._solver = solver
        self._work_dirs = {}   # {role: repertoire de travail XFoil}

    @property
    def work_dirs(self):
        u"""Repertoires de travail XFoil par role (pour le diagnostic).

        Renseigne au fur et a mesure, y compris pour un role dont la
        simulation a echoue (le log reste consultable).

        :rtype: dict
        """
        return dict(self._work_dirs)

    def run(self):
        u"""Execute les simulations (dans le thread)."""
        results = {}
        try:
            todo = [(r, p) for r, p in self._profils.items()
                    if p is not None]
            total = len(todo)
            self.progress_step.emit(0, total)
            for done, (role, profil) in enumerate(todo):
                label = profil.name or role
                self.progress.emit("Simulation '%s'..." % label)
                logger.info("Lancement simulation %s (%s)", role, label)

                normalize = role not in self._no_normalize_roles
                sim = Simulation(profil, params=self._params,
                                 solver=self._solver, normalize=normalize)
                # Memoriser le repertoire AVANT run() pour qu'il reste
                # consultable meme si la simulation echoue.
                self._work_dirs[role] = sim.work_dir
                sim_results = sim.run()
                results[role] = sim_results

                n_pts = sim_results.n_converged
                self.progress.emit(
                    "'%s' : %d points converges" % (label, n_pts))
                self.progress_step.emit(done + 1, total)

            self.finished_ok.emit(results)

        except Exception as e:
            logger.error("Erreur simulation : %s", str(e))
            self.finished_error.emit(str(e))
