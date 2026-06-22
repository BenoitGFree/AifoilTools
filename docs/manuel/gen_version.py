#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genere docs/manuel/version.tex depuis __version__ (source unique).

Ecrit la commande LaTeX ``\\manuelversion`` a partir de la valeur de
``sources/gui/__init__.py``. Appele automatiquement par
``AirfoilTools.spec`` au build, et lancable a la main avant une
compilation LaTeX isolee :

    python docs/manuel/gen_version.py

Le fichier produit (``version.tex``) est un artefact : il n'est pas
suivi par Git (voir .gitignore).
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
_INIT = os.path.join(_ROOT, 'sources', 'gui', '__init__.py')
_OUT = os.path.join(_HERE, 'version.tex')


def read_version():
    """Lit ``__version__`` depuis sources/gui/__init__.py sans l'importer.

    :returns: la chaine de version (ex: ``"3.0"``)
    :rtype: str
    :raises RuntimeError: si ``__version__`` est introuvable
    """
    with open(_INIT, encoding='utf-8') as fh:
        match = re.search(
            r'''^__version__\s*=\s*['"]([^'"]+)['"]''',
            fh.read(), re.MULTILINE)
    if match is None:
        raise RuntimeError(u"__version__ introuvable dans %s" % _INIT)
    return match.group(1)


def generate():
    """(Re)genere docs/manuel/version.tex depuis ``__version__``.

    :returns: la version ecrite
    :rtype: str
    """
    version = read_version()
    content = (
        u"% Fichier GENERE automatiquement (docs/manuel/gen_version.py).\n"
        u"% Source : sources/gui/__init__.py (__version__). "
        u"Ne pas editer a la main.\n"
        u"\\newcommand{\\manuelversion}{" + version + u"}\n")
    with open(_OUT, 'w', encoding='utf-8') as fh:
        fh.write(content)
    return version


if __name__ == '__main__':
    v = generate()
    sys.stdout.write(u"version.tex genere : manuelversion = %s\n" % v)
