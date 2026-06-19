#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Sauvegarde et chargement de projets AirfoilTools (.aftproj).

Un projet regroupe dans un fichier JSON :

- le profil courant et le profil de reference (mode discret ou spline),
- une image de calque (a decalquer) embarquee en base64, avec sa
  transformation (position, echelle, rotation, visibilite).

Le format JSON rend le projet portable : l'image est embarquee, donc on
la retrouve a la reouverture meme si le fichier image d'origine a ete
deplace ou supprime.

Usage::

    from model.project import save_project, load_project, encode_image_array

    img = {
        'filename': 'calque.jpg',
        'data_b64': encode_image_array(arr),
        'px': 500.0, 'py': 0.0, 'scale': 1.0, 'angle': 0.0,
        'visible': True,
    }
    save_project('mon_profil.aftproj', current, reference, image=img)
    current, reference, img = load_project('mon_profil.aftproj')

@author: Nervures
@date: 2026-06
"""

import os
import json
import base64
import logging
from io import BytesIO

import numpy as np

if __name__ == '__main__' and not __package__:
    import sys as _sys
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from model.profil_spline import ProfilSpline
    from model.bezier import Bezier
    from model.bezier_spline import BezierSpline
else:
    from .profil_spline import ProfilSpline
    from .bezier import Bezier
    from .bezier_spline import BezierSpline

logger = logging.getLogger(__name__)

PROJECT_VERSION = 1
PROJECT_EXT = '.aftproj'


# ----------------------------------------------------------------------
#  Serialisation des profils
# ----------------------------------------------------------------------

def _spline_to_dict(spl):
    u"""Serialise une BezierSpline en dict JSON.

    :param spl: spline a serialiser
    :type spl: BezierSpline
    :returns: dict serialisable
    :rtype: dict
    """
    return {
        'segments': [seg.control_points.tolist() for seg in spl.segments],
        'continuities': spl.continuities,
        'n_points': spl.n_points,
        'sample_mode': spl.sample_mode,
        'ba_vertical': getattr(spl, 'ba_vertical', True),
    }


def _spline_from_dict(d, name):
    u"""Reconstruit une BezierSpline depuis un dict JSON.

    :param d: dict produit par :func:`_spline_to_dict`
    :type d: dict
    :param name: nom de la spline
    :type name: str
    :returns: spline reconstruite
    :rtype: BezierSpline
    """
    segments = [Bezier(np.array(cp, dtype=float)) for cp in d['segments']]
    continuities = d.get('continuities') or None
    spl = BezierSpline(
        segments, continuities=continuities, name=name,
        n_points=d.get('n_points', 100),
        sample_mode=d.get('sample_mode', 'curvilinear'))
    spl.ba_vertical = d.get('ba_vertical', True)
    return spl


def profil_to_dict(p):
    u"""Serialise un ProfilSpline en dict JSON.

    :param p: profil a serialiser (ou None)
    :type p: ProfilSpline or None
    :returns: dict serialisable, ou None
    :rtype: dict or None
    """
    if p is None:
        return None
    d = {'name': p.name, 'points': p.points.tolist()}
    if p.has_splines:
        d['splines'] = {
            'extrados': _spline_to_dict(p.spline_extrados),
            'intrados': _spline_to_dict(p.spline_intrados),
        }
    return d


def profil_from_dict(d):
    u"""Reconstruit un ProfilSpline depuis un dict JSON.

    :param d: dict produit par :func:`profil_to_dict` (ou None)
    :type d: dict or None
    :returns: profil reconstruit, ou None
    :rtype: ProfilSpline or None
    """
    if d is None:
        return None
    name = d.get('name', 'Sans nom')
    p = ProfilSpline(np.array(d['points'], dtype=float), name=name)
    splines = d.get('splines')
    if splines:
        p._spline_extrados = _spline_from_dict(
            splines['extrados'], '%s extrados' % name)
        p._spline_intrados = _spline_from_dict(
            splines['intrados'], '%s intrados' % name)
    return p


# ----------------------------------------------------------------------
#  Image de calque : encodage / decodage base64
# ----------------------------------------------------------------------

def encode_image_array(arr):
    u"""Encode un tableau image en PNG base64 (sans perte).

    :param arr: image RGB(A), ndarray(h, w, 3|4)
    :type arr: numpy.ndarray
    :returns: chaine base64 (ASCII)
    :rtype: str
    """
    from PIL import Image
    im = Image.fromarray(np.asarray(arr).astype('uint8'))
    buf = BytesIO()
    im.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def decode_image_b64(data_b64):
    u"""Decode une image base64 en tableau RGBA.

    :param data_b64: chaine base64 produite par :func:`encode_image_array`
    :type data_b64: str
    :returns: image RGBA, ndarray(h, w, 4)
    :rtype: numpy.ndarray
    """
    from PIL import Image
    raw = base64.b64decode(data_b64)
    with Image.open(BytesIO(raw)) as im:
        return np.asarray(im.convert('RGBA'))


# ----------------------------------------------------------------------
#  Sauvegarde / chargement du projet
# ----------------------------------------------------------------------

def save_project(filepath, current, reference, image=None):
    u"""Sauvegarde un projet AirfoilTools au format JSON (.aftproj).

    :param filepath: chemin du fichier de sortie
    :type filepath: str
    :param current: profil courant (ou None)
    :type current: ProfilSpline or None
    :param reference: profil de reference (ou None)
    :type reference: ProfilSpline or None
    :param image: dict de l'image de calque (cles : filename, data_b64,
        px, py, scale, angle, visible), ou None
    :type image: dict or None
    :returns: chemin du fichier ecrit
    :rtype: str
    """
    data = {
        'version': PROJECT_VERSION,
        'application': 'AirfoilTools',
        'current': profil_to_dict(current),
        'reference': profil_to_dict(reference),
        'image': image,
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    logger.info(u"Projet sauvegarde dans %s", filepath)
    return filepath


def load_project(filepath):
    u"""Charge un projet AirfoilTools (.aftproj).

    :param filepath: chemin du fichier projet
    :type filepath: str
    :returns: (profil_courant, profil_reference, image_dict)
    :rtype: tuple(ProfilSpline or None, ProfilSpline or None, dict or None)
    :raises ValueError: si le fichier n'est pas un projet valide
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict) or 'version' not in data:
        raise ValueError(
            u"Fichier projet invalide : %s" % filepath)

    current = profil_from_dict(data.get('current'))
    reference = profil_from_dict(data.get('reference'))
    image = data.get('image')
    return current, reference, image
