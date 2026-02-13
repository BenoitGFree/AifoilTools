#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Simulateur XFoil : execute xfoil.exe avec un fichier de commandes.

Utilise subprocess pour lancer XFoil et rediriger le fichier de commandes
en stdin. Gere le timeout et les erreurs d'execution.

@author: Nervures
@date: 2026-02
"""

import os
import subprocess
import logging

from .base import AbstractSimulator

logger = logging.getLogger(__name__)


class XFoilSimulator(AbstractSimulator):
    u"""Execute XFoil avec un fichier de commandes.

    L'executable XFoil est cherche dans externaltools/xfoil/ par defaut.
    Le fichier de commandes genere par le preprocesseur est passe en stdin.
    """

    def __init__(self, exe_path=None, timeout=30):
        u"""
        :param exe_path: chemin vers xfoil.exe (None = auto-detection au run)
        :type exe_path: str or None
        :param timeout: timeout en secondes
        :type timeout: int
        """
        if exe_path is None:
            # Detection lazy : on cherche maintenant mais sans erreur fatale
            try:
                exe_path = self._find_xfoil()
            except IOError:
                exe_path = 'xfoil'  # sera verifie au run()
                logger.warning(u"XFoil non detecte a l'initialisation. "
                               u"Sera cherche au moment du run().")
        super(XFoilSimulator, self).__init__(exe_path, timeout)

    def _find_xfoil(self):
        u"""Cherche l'executable XFoil dans les emplacements standards.

        Ordre de recherche :
        1. externaltools/xfoil/xfoil.exe (relatif au projet)
        2. PATH systeme

        :returns: chemin de l'executable
        :rtype: str
        :raises IOError: si XFoil n'est pas trouve
        """
        # Chercher dans externaltools/
        base = os.path.dirname(os.path.abspath(__file__))
        # Remonter de sources/model/ vers la racine du projet
        project_root = os.path.normpath(
            os.path.join(base, '..', '..'))
        candidates = [
            os.path.join(project_root, 'externaltools', 'xfoil', 'xfoil.exe'),
            os.path.join(project_root, 'externaltools', 'xfoil', 'xfoil'),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path

        # Chercher dans le PATH
        for dir_path in os.environ.get('PATH', '').split(os.pathsep):
            for name in ('xfoil.exe', 'xfoil'):
                candidate = os.path.join(dir_path, name)
                if os.path.isfile(candidate):
                    return candidate

        raise IOError(
            u"XFoil introuvable. Placez xfoil.exe dans "
            u"externaltools/xfoil/ ou ajoutez-le au PATH."
        )

    def run(self, work_dir, input_files):
        u"""Lance XFoil avec le fichier de commandes.

        :param work_dir: repertoire de travail
        :type work_dir: str
        :param input_files: fichiers generes par le preprocesseur
        :type input_files: list[str]
        :returns: True si l'execution s'est terminee normalement
        :rtype: bool
        """
        # Trouver le fichier de commandes (.cmd) dans input_files
        cmd_files = [f for f in input_files
                     if f.endswith('.cmd')]

        success = True
        for cmd_file in cmd_files:
            ok = self._run_single(work_dir, cmd_file)
            if not ok:
                success = False

        return success

    def _run_single(self, work_dir, cmd_file):
        u"""Execute XFoil avec un seul fichier de commandes.

        :param work_dir: repertoire de travail
        :type work_dir: str
        :param cmd_file: chemin du fichier de commandes
        :type cmd_file: str
        :returns: True si OK
        :rtype: bool
        """
        if not os.path.isfile(self.exe_path):
            logger.error(u"Executable XFoil introuvable : %s", self.exe_path)
            return False

        if not os.path.isfile(cmd_file):
            logger.error(u"Fichier de commandes introuvable : %s", cmd_file)
            return False

        logger.info(u"Execution XFoil : %s < %s", self.exe_path, cmd_file)
        logger.info(u"  Repertoire de travail : %s", work_dir)
        logger.info(u"  Timeout : %d s", self.timeout)

        try:
            with open(cmd_file, 'r') as f_in:
                cmd_content = f_in.read()

            proc = subprocess.Popen(
                [self.exe_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=work_dir
            )

            try:
                stdout, stderr = proc.communicate(
                    input=cmd_content.encode('utf-8'),
                    timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                logger.error(
                    u"XFoil timeout (%d s) pour %s", self.timeout, cmd_file)
                return False

            stdout = stdout.decode('utf-8', errors='replace') if stdout else ''
            stderr = stderr.decode('utf-8', errors='replace') if stderr else ''

            # Sauvegarder la sortie console de XFoil pour debug
            log_file = os.path.join(work_dir,
                                    os.path.basename(cmd_file) + '.log')
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(stdout)
                if stderr:
                    f.write('\n--- STDERR ---\n')
                    f.write(stderr)

            if proc.returncode != 0:
                logger.warning(
                    u"XFoil retourne code %d pour %s",
                    proc.returncode, cmd_file)
                return False

            logger.info(u"XFoil termine avec succes")
            return True

        except OSError as e:
            logger.error(u"Erreur d'execution XFoil : %s", str(e))
            return False
        except Exception as e:
            logger.error(u"Erreur inattendue XFoil : %s", str(e))
            return False
