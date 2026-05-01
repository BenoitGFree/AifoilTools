#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Acces a la base de profils UIUC (Selig).

Telechargement et mise en cache de l'index et des fichiers .dat depuis
https://m-selig.ae.illinois.edu/ads/coord_database.html
"""

import json
import logging
import os
import re
import time
import urllib.request
from collections import namedtuple
from pathlib import Path

logger = logging.getLogger(__name__)

UIUC_BASE = "https://m-selig.ae.illinois.edu/ads"
UIUC_INDEX_URL = "%s/coord_database.html" % UIUC_BASE

INDEX_TTL_SECONDS = 7 * 24 * 3600  # 7 jours
HTTP_TIMEOUT = 20

# Regex : <a href="coord/NAME.dat">NAME.dat</a> \ DESCRIPTION
# La description s'arrete au prochain <a, <br ou fin de ligne
_RE_ENTRY = re.compile(
    r'<a\s+href="(coord(?:_updates)?/([^"]+\.dat))"[^>]*>\s*([^<]+?)\s*</a>'
    r'\s*\\\s*([^<\n\r]*?)(?=\s*\\\s*<a|\s*<br|\s*</p>|\s*$)',
    re.IGNORECASE)


UIUCEntry = namedtuple('UIUCEntry', ['name', 'description', 'dat_url'])
u"""Une entree de la base UIUC.

:ivar name: nom du fichier (ex: 'naca2412.dat')
:ivar description: description courte ou chaine vide
:ivar dat_url: URL absolue du fichier .dat
"""


def default_cache_dir():
    u"""Repertoire de cache par defaut : ~/.airfoiltools/cache/uiuc/."""
    return Path.home() / '.airfoiltools' / 'cache' / 'uiuc'


class UIUCLoader(object):
    u"""Charge la liste des profils UIUC et les fichiers .dat.

    Utilise un cache local pour eviter de retelecharger inutilement.
    L'index est mis en cache pour ``INDEX_TTL_SECONDS`` ; les fichiers
    .dat sont mis en cache de maniere permanente (les fichiers UIUC
    sont stables dans le temps).

    :param cache_dir: repertoire de cache (defaut : ~/.airfoiltools/cache/uiuc/)
    :type cache_dir: pathlib.Path or str or None
    """

    def __init__(self, cache_dir=None):
        if cache_dir is None:
            cache_dir = default_cache_dir()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_cache_file = self.cache_dir / 'index.json'

    # ------------------------------------------------------------------
    #  Index
    # ------------------------------------------------------------------

    def fetch_index(self, force=False):
        u"""Retourne la liste des profils disponibles.

        Le resultat est mis en cache pour ``INDEX_TTL_SECONDS`` (7 jours).

        :param force: si True, ignore le cache et retelecharge
        :type force: bool
        :returns: liste d'entrees triees par nom
        :rtype: list[UIUCEntry]
        :raises urllib.error.URLError: en cas d'echec reseau et absence
            de cache exploitable
        """
        if not force:
            cached = self._load_cached_index()
            if cached is not None:
                logger.debug(
                    u"Index UIUC charge depuis le cache (%d entrees)",
                    len(cached))
                return cached

        logger.info(u"Telechargement de l'index UIUC...")
        html = self._http_get_text(UIUC_INDEX_URL)
        entries = parse_index_html(html)
        self._save_cached_index(entries)
        logger.info(u"Index UIUC : %d profils", len(entries))
        return entries

    def _load_cached_index(self):
        u"""Charge l'index depuis le cache si valide, sinon None."""
        if not self._index_cache_file.exists():
            return None
        try:
            with open(self._index_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        ts = data.get('timestamp', 0)
        if time.time() - ts > INDEX_TTL_SECONDS:
            return None
        entries = [UIUCEntry(**e) for e in data.get('entries', [])]
        return entries if entries else None

    def _save_cached_index(self, entries):
        u"""Sauve l'index dans le cache JSON."""
        data = {
            'timestamp': time.time(),
            'entries': [e._asdict() for e in entries],
        }
        try:
            with open(self._index_cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning(
                u"Impossible d'ecrire le cache index : %s", exc)

    # ------------------------------------------------------------------
    #  Telechargement d'un .dat
    # ------------------------------------------------------------------

    def fetch_dat(self, entry, force=False):
        u"""Telecharge le fichier .dat d'un profil et retourne son chemin local.

        Le fichier est mis en cache de maniere permanente.

        :param entry: entree du profil
        :type entry: UIUCEntry
        :param force: si True, ignore le cache et retelecharge
        :type force: bool
        :returns: chemin local du fichier .dat
        :rtype: pathlib.Path
        :raises urllib.error.URLError: en cas d'echec reseau
        """
        local = self.cache_dir / entry.name
        if local.exists() and not force:
            return local

        logger.info(u"Telechargement de %s...", entry.dat_url)
        content = self._http_get_bytes(entry.dat_url)
        local.write_bytes(content)
        return local

    # ------------------------------------------------------------------
    #  HTTP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _http_get_text(url):
        u"""GET d'une ressource texte (UTF-8)."""
        req = urllib.request.Request(
            url, headers={'User-Agent': 'AirfoilTools/2.0'})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.read().decode('utf-8', errors='replace')

    @staticmethod
    def _http_get_bytes(url):
        u"""GET d'une ressource binaire."""
        req = urllib.request.Request(
            url, headers={'User-Agent': 'AirfoilTools/2.0'})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.read()


# ----------------------------------------------------------------------
#  Parsing (fonction module-level pour testabilite)
# ----------------------------------------------------------------------

def parse_index_html(html):
    u"""Extrait la liste des profils a partir du HTML de l'index UIUC.

    :param html: contenu HTML de la page coord_database.html
    :type html: str
    :returns: liste d'entrees triees par nom
    :rtype: list[UIUCEntry]
    """
    seen = set()
    entries = []
    for match in _RE_ENTRY.finditer(html):
        rel_path, name, _link_text, description = match.groups()
        if name in seen:
            continue
        seen.add(name)
        url = "%s/%s" % (UIUC_BASE, rel_path)
        description = description.strip()
        # Nettoyer eventuels retours a la ligne ou tabulations
        description = re.sub(r'\s+', ' ', description)
        entries.append(UIUCEntry(name=name,
                                 description=description,
                                 dat_url=url))
    entries.sort(key=lambda e: e.name.lower())
    return entries
