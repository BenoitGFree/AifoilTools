#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Classes abstraites pour l'analyse aerodynamique 2D de profils.

Architecture extensible : chaque solveur (XFoil, JavaFoil, MSES...)
implemente les 3 classes abstraites Preprocessor, Simulator, Postprocessor.

@author: Nervures
@date: 2026-02
"""

from abc import ABC, abstractmethod


class AbstractPreprocessor(ABC):
    u"""Prepare les donnees d'entree pour un solveur 2D.

    Responsabilites :
    - Recevoir les points du profil et les parametres aerodynamiques
    - Generer les fichiers d'entree specifiques au solveur
    - Gerer le repertoire de travail
    """

    def __init__(self, work_dir):
        u"""
        :param work_dir: repertoire de travail pour les fichiers generes
        :type work_dir: str
        """
        self.work_dir = work_dir

    @abstractmethod
    def prepare(self, profile_points, params):
        u"""Genere les fichiers d'entree du solveur.

        :param profile_points: coordonnees du profil, shape (n, 2)
        :type profile_points: numpy.ndarray
        :param params: parametres de simulation
        :type params: dict
        :returns: liste des fichiers generes
        :rtype: list[str]
        """
        pass


class AbstractSimulator(ABC):
    u"""Execute un solveur aerodynamique 2D.

    Responsabilites :
    - Localiser l'executable du solveur
    - Lancer le calcul avec timeout
    - Gerer les erreurs d'execution
    """

    def __init__(self, exe_path, timeout=30):
        u"""
        :param exe_path: chemin vers l'executable du solveur
        :type exe_path: str
        :param timeout: timeout en secondes
        :type timeout: int
        """
        self.exe_path = exe_path
        self.timeout = timeout

    @abstractmethod
    def run(self, work_dir, input_files):
        u"""Lance le solveur.

        :param work_dir: repertoire de travail
        :type work_dir: str
        :param input_files: fichiers d'entree generes par le preprocessor
        :type input_files: list[str]
        :returns: True si le calcul a converge, False sinon
        :rtype: bool
        """
        pass


class AbstractPostprocessor(ABC):
    u"""Parse les resultats d'un solveur aerodynamique 2D.

    Responsabilites :
    - Lire les fichiers de sortie du solveur
    - Structurer les resultats en donnees neutres (dicts + numpy arrays)
    """

    @abstractmethod
    def parse(self, work_dir):
        u"""Lit et structure les resultats du solveur.

        :param work_dir: repertoire contenant les fichiers de sortie
        :type work_dir: str
        :returns: resultats structures
        :rtype: dict

        Le dictionnaire retourne doit contenir au minimum :
        - 'polar' : dict avec cles 'alpha', 'CL', 'CD', 'CDp', 'CM',
                     'Top_Xtr', 'Bot_Xtr' (numpy arrays)
        - 'cp'    : dict alpha -> ndarray(n, 2) [x, Cp] (si disponible)
        - 'bl'    : dict alpha -> donnees couche limite (si disponible)
        - 'warnings' : liste de messages d'avertissement
        """
        pass
