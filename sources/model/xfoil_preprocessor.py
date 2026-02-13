#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Preprocesseur XFoil : genere les fichiers d'entree pour XFoil.

- Fichier profil (.dat) au format Selig (x y, une ligne par point)
- Fichier de commandes (.cmd) pour piloter XFoil

@author: Nervures
@date: 2026-02
"""

import os
import numpy as np

from .base import AbstractPreprocessor
from .foilconfig import load_defaults, merge_params


def _re_label(re_val):
    u"""Formate un Reynolds en label lisible pour les noms de fichiers.

    Evite la notation scientifique (1e+06 -> 1000000).

    :param re_val: nombre de Reynolds
    :type re_val: float or int
    :returns: label (ex: '300000', '1000000')
    :rtype: str
    """
    return '%d' % int(re_val)


class XFoilPreprocessor(AbstractPreprocessor):
    u"""Genere les fichiers d'entree pour XFoil.

    Cree dans work_dir :
    - profil.dat : coordonnees du profil
    - xfoil_alpha.cmd : script de commandes pour polaire alpha
    - xfoil_cl.cmd : script de commandes pour polaire Cl fixe (si demande)
    """

    def __init__(self, work_dir):
        super(XFoilPreprocessor, self).__init__(work_dir)

    def prepare(self, profile_points, params=None):
        u"""Genere tous les fichiers d'entree XFoil.

        :param profile_points: coordonnees du profil, shape (n, 2)
        :type profile_points: numpy.ndarray
        :param params: parametres utilisateur (surchargent les defauts)
        :type params: dict or None
        :returns: liste des fichiers generes
        :rtype: list[str]
        """
        defaults = load_defaults('xfoil')
        self.params = merge_params(defaults, params)

        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)

        generated = []

        # Ecriture du profil
        profil_file = self._write_profile(profile_points)
        generated.append(profil_file)

        # Scripts de commande selon le mode d'analyse
        re_list = self.params.get('RE_LIST', [self.params['RE']])
        if not isinstance(re_list, (list, tuple)):
            re_list = [re_list]

        # Polaire alpha
        cmd_file = self._write_alpha_commands(re_list)
        generated.append(cmd_file)

        # Polaire Cl fixe (si demande)
        if 'CL_TARGET' in self.params or 'CL_LIST' in self.params:
            cmd_file = self._write_cl_commands(re_list)
            generated.append(cmd_file)

        return generated

    def _write_profile(self, points):
        u"""Ecrit le fichier profil au format Selig (.dat).

        :param points: coordonnees (n, 2)
        :type points: numpy.ndarray
        :returns: chemin du fichier cree
        :rtype: str
        """
        filepath = os.path.join(self.work_dir, 'profil.dat')
        pts = np.asarray(points, dtype=float)
        with open(filepath, 'w') as f:
            f.write('Profil Axile\n')
            for i in range(pts.shape[0]):
                f.write(' %10.6f %10.6f\n' % (pts[i, 0], pts[i, 1]))
        return filepath

    def _write_alpha_commands(self, re_list):
        u"""Genere le script XFoil pour un balayage en alpha.

        Pour chaque Re, genere :
        - un fichier polaire : polar_Re<value>.dat
        - un fichier Cp : cp_Re<value>_a<alpha>.dat (pour chaque alpha)
        - un fichier BL : bl_Re<value>.dat

        Gestion multi-Re : on reste dans OPER et on toggle VISC OFF/ON
        entre chaque Re, avec INIT pour reinitialiser la couche limite.

        :param re_list: liste des nombres de Reynolds
        :type re_list: list
        :returns: chemin du fichier commande
        :rtype: str
        """
        p = self.params
        filepath = os.path.join(self.work_dir, 'xfoil_alpha.cmd')
        lines = []

        # Chargement du profil
        lines.append('LOAD profil.dat')
        lines.append('Profil_Axile')

        # Repaneling si demande
        if p.get('REPANEL', True):
            lines.append('PPAR')
            lines.append('N %d' % p.get('NPANEL', 200))
            lines.append('')  # ligne vide pour valider PPAR
            lines.append('')

        # Entrer dans OPER une seule fois
        lines.append('OPER')

        alpha_min = p.get('ALPHA_MIN', -5.0)
        alpha_max = p.get('ALPHA_MAX', 15.0)
        alpha_step = p.get('ALPHA_STEP', 0.5)

        for i_re, re_val in enumerate(re_list):
            re_tag = _re_label(re_val)

            # Pour le 2e Re et suivants : toggle VISC OFF puis ON
            if p.get('VISCOUS', True):
                if i_re > 0:
                    lines.append('VISC')  # toggle OFF
                    lines.append('INIT')  # reinitialiser la BL
                lines.append('VISC %g' % re_val)  # toggle ON avec Re
                lines.append('MACH %g' % p.get('MACH', 0.0))
                # Transition forcee
                lines.append('VPAR')
                lines.append('XTR %g %g' % (p.get('XTR_TOP', 0.01),
                                             p.get('XTR_BOT', 0.01)))
                lines.append('N %g' % p.get('NCRIT', 9))
                lines.append('')  # sortir de VPAR

            # Nombre max d'iterations
            lines.append('ITER %d' % p.get('ITER', 100))

            # Fichier de sortie polaire
            polar_file = 'polar_Re%s.dat' % re_tag
            lines.append('PACC')
            lines.append(polar_file)
            lines.append('')  # pas de dump file

            # Balayage alpha (polaire)
            lines.append('ASEQ %g %g %g' % (alpha_min, alpha_max, alpha_step))

            # Desactiver l'accumulation polaire
            lines.append('PACC')

            # Sauvegarde Cp pour chaque alpha
            alpha = alpha_min
            while alpha <= alpha_max + 1e-9:
                lines.append('ALFA %g' % alpha)
                cp_file = 'cp_Re%s_a%g.dat' % (re_tag, alpha)
                lines.append('CPWR %s' % cp_file)
                alpha += alpha_step

            # Sauvegarde couche limite pour le dernier alpha
            lines.append('DUMP bl_Re%s.dat' % re_tag)

        # Sortir de OPER et quitter
        lines.append('')
        lines.append('QUIT')

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        return filepath

    def _write_cl_commands(self, re_list):
        u"""Genere le script XFoil pour une analyse a Cl fixe.

        :param re_list: liste des nombres de Reynolds
        :type re_list: list
        :returns: chemin du fichier commande
        :rtype: str
        """
        p = self.params
        filepath = os.path.join(self.work_dir, 'xfoil_cl.cmd')
        lines = []

        cl_list = p.get('CL_LIST', [p.get('CL_TARGET', 0.5)])
        if not isinstance(cl_list, (list, tuple)):
            cl_list = [cl_list]

        # Chargement du profil
        lines.append('LOAD profil.dat')
        lines.append('Profil_Axile')

        if p.get('REPANEL', True):
            lines.append('PPAR')
            lines.append('N %d' % p.get('NPANEL', 200))
            lines.append('')
            lines.append('')

        lines.append('OPER')

        for i_re, re_val in enumerate(re_list):
            re_tag = _re_label(re_val)

            if p.get('VISCOUS', True):
                if i_re > 0:
                    lines.append('VISC')  # toggle OFF
                    lines.append('INIT')
                lines.append('VISC %g' % re_val)  # toggle ON
                lines.append('MACH %g' % p.get('MACH', 0.0))
                lines.append('VPAR')
                lines.append('XTR %g %g' % (p.get('XTR_TOP', 0.01),
                                             p.get('XTR_BOT', 0.01)))
                lines.append('N %g' % p.get('NCRIT', 9))
                lines.append('')

            lines.append('ITER %d' % p.get('ITER', 100))

            polar_file = 'polar_cl_Re%s.dat' % re_tag
            lines.append('PACC')
            lines.append(polar_file)
            lines.append('')

            for cl_val in cl_list:
                lines.append('CL %g' % cl_val)

            lines.append('PACC')

        lines.append('')
        lines.append('QUIT')

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        return filepath
