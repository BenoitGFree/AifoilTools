#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Pipeline d'analyse aerodynamique 2D.

Orchestre les 3 etapes : Preprocessor -> Simulator -> Postprocessor.
Agnostique du solveur : instancie les bonnes classes selon le parametre solver.

Usage::

    from model.aerodynamique.foil2d.pipeline import FoilAnalysisPipeline

    pipeline = FoilAnalysisPipeline(solver='xfoil')
    results = pipeline.run(profile_points, params)

@author: Nervures
@date: 2026-02
"""

import os
import tempfile
import logging

logger = logging.getLogger(__name__)

# Registre des solveurs disponibles
# Chaque entree : 'nom' -> (module_pre, class_pre, module_sim, class_sim, module_post, class_post)
_SOLVER_REGISTRY = {}


def register_solver(name, preprocessor_cls, simulator_cls, postprocessor_cls):
    u"""Enregistre un nouveau solveur dans le registre.

    :param name: nom du solveur (ex: 'xfoil')
    :type name: str
    :param preprocessor_cls: classe du preprocesseur
    :param simulator_cls: classe du simulateur
    :param postprocessor_cls: classe du postprocesseur
    """
    _SOLVER_REGISTRY[name] = (preprocessor_cls, simulator_cls,
                              postprocessor_cls)


def _ensure_xfoil_registered():
    u"""Enregistre XFoil si ce n'est pas deja fait (lazy import)."""
    if 'xfoil' not in _SOLVER_REGISTRY:
        from xfoil_preprocessor import XFoilPreprocessor
        from xfoil_simulator import XFoilSimulator
        from xfoil_postprocessor import XFoilPostprocessor
        register_solver('xfoil', XFoilPreprocessor, XFoilSimulator,
                        XFoilPostprocessor)


class FoilAnalysisPipeline(object):
    u"""Orchestre une analyse aerodynamique 2D complete.

    Enchaine automatiquement :
    1. Preprocessing (generation des fichiers d'entree)
    2. Simulation (execution du solveur)
    3. Postprocessing (parsing des resultats)

    Les etapes peuvent aussi etre appelees individuellement.
    """

    def __init__(self, solver='xfoil', work_dir=None, exe_path=None):
        u"""
        :param solver: nom du solveur ('xfoil', etc.)
        :type solver: str
        :param work_dir: repertoire de travail (None = temporaire)
        :type work_dir: str or None
        :param exe_path: chemin de l'executable (None = auto)
        :type exe_path: str or None
        """
        _ensure_xfoil_registered()

        if solver not in _SOLVER_REGISTRY:
            raise ValueError(
                u"Solveur '%s' inconnu. Disponibles : %s"
                % (solver, ', '.join(sorted(_SOLVER_REGISTRY.keys()))))

        pre_cls, sim_cls, post_cls = _SOLVER_REGISTRY[solver]

        self.solver_name = solver
        self._work_dir_provided = work_dir is not None
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix='foil2d_')
        self.work_dir = work_dir

        self.preprocessor = pre_cls(work_dir)
        self.simulator = sim_cls(exe_path=exe_path)
        self.postprocessor = post_cls()

        self.results = None

    def run(self, profile_points, params=None):
        u"""Execute le pipeline complet : Pre -> Sim -> Post.

        :param profile_points: coordonnees du profil, shape (n, 2)
        :type profile_points: numpy.ndarray
        :param params: parametres utilisateur (surchargent les defauts)
        :type params: dict or None
        :returns: resultats structures (voir AbstractPostprocessor.parse)
        :rtype: dict
        """
        logger.info(u"=== Demarrage pipeline foil2d [%s] ===",
                    self.solver_name)
        logger.info(u"  Repertoire de travail : %s", self.work_dir)

        # Propager le timeout des parametres au simulateur
        if params and 'TIMEOUT' in params:
            self.simulator.timeout = int(params['TIMEOUT'])

        # Etape 1 : Preprocessing
        logger.info(u"--- Etape 1/3 : Preprocessing ---")
        input_files = self.preprocessor.prepare(profile_points, params)
        logger.info(u"  %d fichiers generes", len(input_files))

        # Etape 2 : Simulation
        logger.info(u"--- Etape 2/3 : Simulation ---")
        success = self.simulator.run(self.work_dir, input_files)
        if not success:
            logger.warning(u"La simulation a rencontre des problemes")

        # Etape 3 : Postprocessing
        logger.info(u"--- Etape 3/3 : Postprocessing ---")
        self.results = self.postprocessor.parse(self.work_dir)

        n_warnings = len(self.results.get('warnings', []))
        if n_warnings > 0:
            logger.warning(u"  %d avertissements", n_warnings)
            for w in self.results['warnings']:
                logger.warning(u"    %s", w)

        logger.info(u"=== Pipeline termine ===")
        return self.results

    def prepare_only(self, profile_points, params=None):
        u"""Execute uniquement le preprocessing.

        Utile pour inspecter les fichiers generes avant de lancer le calcul.

        :param profile_points: coordonnees du profil
        :type profile_points: numpy.ndarray
        :param params: parametres utilisateur
        :type params: dict or None
        :returns: liste des fichiers generes
        :rtype: list[str]
        """
        return self.preprocessor.prepare(profile_points, params)

    def simulate_only(self, input_files):
        u"""Execute uniquement la simulation.

        :param input_files: fichiers d'entree
        :type input_files: list[str]
        :returns: succes
        :rtype: bool
        """
        return self.simulator.run(self.work_dir, input_files)

    def parse_only(self):
        u"""Execute uniquement le postprocessing.

        :returns: resultats
        :rtype: dict
        """
        self.results = self.postprocessor.parse(self.work_dir)
        return self.results
