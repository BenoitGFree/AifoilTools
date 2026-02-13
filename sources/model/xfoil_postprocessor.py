#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Postprocesseur XFoil : parse les fichiers de sortie XFoil.

Lit les fichiers generes par XFoil :
- polar_Re*.dat : polaires (alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr)
- cp_Re*_a*.dat : distributions de pression Cp(x)
- bl_Re*.dat : donnees de couche limite

@author: Nervures
@date: 2026-02
"""

import os
import re
import logging
import numpy as np

from .base import AbstractPostprocessor

logger = logging.getLogger(__name__)


class XFoilPostprocessor(AbstractPostprocessor):
    u"""Parse les fichiers de sortie XFoil en structures de donnees neutres."""

    def parse(self, work_dir):
        u"""Lit tous les fichiers de resultats dans work_dir.

        :param work_dir: repertoire contenant les fichiers de sortie XFoil
        :type work_dir: str
        :returns: resultats structures
        :rtype: dict

        Structure du dictionnaire retourne::

            {
                'polars': {
                    500000: {
                        'alpha': ndarray,
                        'CL': ndarray,
                        'CD': ndarray,
                        'CDp': ndarray,
                        'CM': ndarray,
                        'Top_Xtr': ndarray,
                        'Bot_Xtr': ndarray
                    },
                    1000000: { ... }
                },
                'cp': {
                    500000: {
                        -5.0: ndarray(n, 2),  # [x, Cp]
                        -4.5: ndarray(n, 2),
                        ...
                    }
                },
                'bl': {
                    500000: ndarray ou None
                },
                'warnings': [str, ...]
            }
        """
        results = {
            'polars': {},
            'cp': {},
            'bl': {},
            'warnings': []
        }

        if not os.path.isdir(work_dir):
            results['warnings'].append(
                u"Repertoire de travail introuvable : %s" % work_dir)
            return results

        files = os.listdir(work_dir)

        # Parse des polaires
        polar_files = sorted(f for f in files
                             if f.startswith('polar_') and f.endswith('.dat')
                             and '_cl_' not in f)
        for pf in polar_files:
            re_val = self._extract_re_from_filename(pf)
            if re_val is not None:
                filepath = os.path.join(work_dir, pf)
                polar_data = self._parse_polar_file(filepath)
                if polar_data is not None:
                    results['polars'][re_val] = polar_data
                else:
                    results['warnings'].append(
                        u"Polaire vide ou illisible : %s" % pf)

        # Parse des polaires Cl fixe
        polar_cl_files = sorted(f for f in files
                                if f.startswith('polar_cl_')
                                and f.endswith('.dat'))
        for pf in polar_cl_files:
            re_val = self._extract_re_from_filename(pf)
            if re_val is not None:
                filepath = os.path.join(work_dir, pf)
                polar_data = self._parse_polar_file(filepath)
                if polar_data is not None:
                    key = 'cl_Re%g' % re_val
                    results['polars'][key] = polar_data

        # Parse des Cp
        cp_files = sorted(f for f in files
                          if f.startswith('cp_') and f.endswith('.dat'))
        for cf in cp_files:
            re_val, alpha = self._extract_re_alpha_from_cp_filename(cf)
            if re_val is not None:
                if re_val not in results['cp']:
                    results['cp'][re_val] = {}
                filepath = os.path.join(work_dir, cf)
                cp_data = self._parse_cp_file(filepath)
                if cp_data is not None:
                    results['cp'][re_val][alpha] = cp_data

        # Parse des couches limites
        bl_files = sorted(f for f in files
                          if f.startswith('bl_') and f.endswith('.dat'))
        for bf in bl_files:
            re_val = self._extract_re_from_filename(bf)
            if re_val is not None:
                filepath = os.path.join(work_dir, bf)
                bl_data = self._parse_bl_file(filepath)
                results['bl'][re_val] = bl_data

        # Bilan
        n_polars = len(results['polars'])
        n_cp = sum(len(v) for v in results['cp'].values())
        logger.info(u"Resultats parses : %d polaires, %d distributions Cp",
                    n_polars, n_cp)

        return results

    def _extract_re_from_filename(self, filename):
        u"""Extrait la valeur de Re depuis un nom de fichier.

        Ex: 'polar_Re500000.dat' -> 500000.0

        :param filename: nom du fichier
        :type filename: str
        :returns: valeur de Re ou None
        :rtype: float or None
        """
        # Retirer l'extension pour eviter que le '.' de '.dat'
        # soit capture dans le nombre
        base = filename.rsplit('.', 1)[0]
        match = re.search(r'Re([0-9.eE+\-]+)', base)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _extract_re_alpha_from_cp_filename(self, filename):
        u"""Extrait Re et alpha depuis un nom de fichier Cp.

        Ex: 'cp_Re500000_a2.5.dat' -> (500000.0, 2.5)
            'cp_Re300000_a-4.5.dat' -> (300000.0, -4.5)

        :param filename: nom du fichier
        :type filename: str
        :returns: (Re, alpha) ou (None, None)
        :rtype: tuple
        """
        # Retirer l'extension pour eviter que le '.' de '.dat'
        # soit capture dans le nombre (ex: 'a0.5.dat' -> 'a0.5.')
        base = filename.rsplit('.', 1)[0]
        match = re.search(
            r'Re([0-9.eE+\-]+)_a([0-9.eE+\-]+)', base)
        if match:
            try:
                re_val = float(match.group(1))
                alpha = float(match.group(2))
                return re_val, alpha
            except ValueError:
                return None, None
        return None, None

    def _parse_polar_file(self, filepath):
        u"""Parse un fichier polaire XFoil.

        Format XFoil (apres 12 lignes d'en-tete) :
        alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr

        :param filepath: chemin du fichier
        :type filepath: str
        :returns: dict de numpy arrays, ou None si vide
        :rtype: dict or None
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        except IOError:
            logger.warning(u"Impossible de lire %s", filepath)
            return None

        # Trouver la ligne de separation '---'
        data_start = 0
        for i, line in enumerate(lines):
            if '----' in line:
                data_start = i + 1
                break

        # Parser les donnees
        data_lines = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    values = [float(x) for x in parts[:7]]
                    data_lines.append(values)
                except ValueError:
                    continue

        if not data_lines:
            return None

        data = np.array(data_lines)
        return {
            'alpha': data[:, 0],
            'CL': data[:, 1],
            'CD': data[:, 2],
            'CDp': data[:, 3],
            'CM': data[:, 4],
            'Top_Xtr': data[:, 5],
            'Bot_Xtr': data[:, 6]
        }

    def _parse_cp_file(self, filepath):
        u"""Parse un fichier de distribution Cp.

        Format XFoil CPWR (3 colonnes) :
        #    x        y        Cp
        (en-tete eventuel, puis une ligne par point)

        :param filepath: chemin du fichier
        :type filepath: str
        :returns: ndarray(n, 2) [x, Cp] ou None
        :rtype: numpy.ndarray or None
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        except IOError:
            return None

        data_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x = float(parts[0])
                    cp = float(parts[2])
                    data_lines.append([x, cp])
                except ValueError:
                    continue
            elif len(parts) == 2:
                try:
                    x = float(parts[0])
                    cp = float(parts[1])
                    data_lines.append([x, cp])
                except ValueError:
                    continue

        if not data_lines:
            return None

        return np.array(data_lines)

    def _parse_bl_file(self, filepath):
        u"""Parse un fichier de couche limite (DUMP).

        Format XFoil DUMP :
        s, x, y, Ue/Vinf, Dstar, Theta, Cf, H
        (avec separateur d'en-tete)

        :param filepath: chemin du fichier
        :type filepath: str
        :returns: dict avec les donnees BL ou None
        :rtype: dict or None
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        except IOError:
            return None

        # Trouver le debut des donnees
        data_start = 0
        for i, line in enumerate(lines):
            if '----' in line or line.strip().startswith('1'):
                # Essayer de parser comme donnees
                parts = line.strip().split()
                if len(parts) >= 8:
                    data_start = i
                    break
                else:
                    data_start = i + 1
                    break

        data_lines = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    values = [float(x) for x in parts[:8]]
                    data_lines.append(values)
                except ValueError:
                    continue

        if not data_lines:
            return None

        data = np.array(data_lines)
        return {
            's': data[:, 0],
            'x': data[:, 1],
            'y': data[:, 2],
            'Ue_Vinf': data[:, 3],
            'Dstar': data[:, 4],
            'Theta': data[:, 5],
            'Cf': data[:, 6],
            'H': data[:, 7]
        }
