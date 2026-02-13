#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Lecture des fichiers de configuration pour les solveurs 2D.

Format : cle=valeur, une par ligne. Les lignes commencant par # sont ignorees.
Les types sont inferes automatiquement (bool, int, float, str).

@author: Nervures
@date: 2026-02
"""

import os


def _parse_value(value_str):
    u"""Infere le type d'une valeur depuis sa representation texte.

    :param value_str: valeur brute lue depuis le fichier
    :type value_str: str
    :returns: valeur typee (bool, int, float ou str)
    """
    s = value_str.strip()
    # Booleens
    if s.lower() in ('true', 'yes', 'on'):
        return True
    if s.lower() in ('false', 'no', 'off'):
        return False
    # Entier
    try:
        return int(s)
    except ValueError:
        pass
    # Flottant
    try:
        return float(s)
    except ValueError:
        pass
    # Chaine
    return s


def load_config(filepath):
    u"""Charge un fichier de configuration cle=valeur.

    :param filepath: chemin du fichier .cfg
    :type filepath: str
    :returns: dictionnaire des parametres
    :rtype: dict
    :raises IOError: si le fichier n'existe pas
    """
    if not os.path.isfile(filepath):
        raise IOError(u"Fichier de configuration introuvable : %s" % filepath)
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            params[key.strip()] = _parse_value(value)
    return params


def load_defaults(solver_name='xfoil'):
    u"""Charge les parametres par defaut d'un solveur.

    Cherche le fichier defaults_<solver_name>.cfg dans le meme repertoire.

    :param solver_name: nom du solveur (ex: 'xfoil')
    :type solver_name: str
    :returns: parametres par defaut
    :rtype: dict
    """
    cfg_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(cfg_dir, 'defaults_%s.cfg' % solver_name)
    return load_config(cfg_file)


def merge_params(defaults, user_params):
    u"""Fusionne les parametres utilisateur avec les defauts.

    Les parametres utilisateur surchargent les defauts.

    :param defaults: parametres par defaut
    :type defaults: dict
    :param user_params: parametres utilisateur (peuvent etre None)
    :type user_params: dict or None
    :returns: parametres fusionnes
    :rtype: dict
    """
    merged = dict(defaults)
    if user_params:
        for key, value in user_params.items():
            merged[key] = value
    return merged
