#!/usr/bin/env python3
# -*- coding: utf-8 -*-
u"""Tests pour le module uiuc_loader."""

import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_here, '..', 'sources'))
if _src not in sys.path:
    sys.path.insert(0, _src)

from model.uiuc_loader import (
    UIUCLoader, UIUCEntry, parse_index_html, INDEX_TTL_SECONDS,
    UIUC_BASE)


# Echantillon HTML representatif (format reel UIUC)
_HTML_SAMPLE = """
<html><body>
<h3>A-Z Directory</h3>
<a href="coord/2032c.dat">2032c.dat</a> \\ Dillner 20-32-C airfoil \\ <a href="afplots/2032c.gif">2032c.gif</a>\\ \\ \\ <br>
<a href="coord/naca2412.dat">naca2412.dat</a> \\ NACA 2412 airfoil \\ <a href="afplots/naca2412.gif">naca2412.gif</a><br>
<a href="coord_updates/bacnlf.dat">bacnlf.dat</a> \\ Boeing high speed natural laminar flow airfoil<br>
<a href="coord_updates/du84132v.dat">du84132v.dat</a> \\ Delft University DU 84-132V airfoil<br>
</body></html>
"""


class TestParseIndexHtml(unittest.TestCase):
    u"""Tests du parsing HTML."""

    def test_parses_coord_entries(self):
        u"""Les entrees coord/ sont extraites."""
        entries = parse_index_html(_HTML_SAMPLE)
        names = [e.name for e in entries]
        self.assertIn('2032c.dat', names)
        self.assertIn('naca2412.dat', names)

    def test_parses_coord_updates_entries(self):
        u"""Les entrees coord_updates/ sont extraites."""
        entries = parse_index_html(_HTML_SAMPLE)
        names = [e.name for e in entries]
        self.assertIn('bacnlf.dat', names)
        self.assertIn('du84132v.dat', names)

    def test_url_absolute(self):
        u"""dat_url est une URL absolue."""
        entries = parse_index_html(_HTML_SAMPLE)
        for e in entries:
            self.assertTrue(
                e.dat_url.startswith(UIUC_BASE),
                u"URL non absolue : %s" % e.dat_url)
            self.assertTrue(e.dat_url.endswith('.dat'))

    def test_description_extracted(self):
        u"""La description est extraite et nettoyee."""
        entries = parse_index_html(_HTML_SAMPLE)
        d = {e.name: e.description for e in entries}
        self.assertIn('NACA 2412', d['naca2412.dat'])
        self.assertIn('Boeing', d['bacnlf.dat'])

    def test_sorted_by_name(self):
        u"""Les entrees sont triees par nom (case-insensitive)."""
        entries = parse_index_html(_HTML_SAMPLE)
        names = [e.name.lower() for e in entries]
        self.assertEqual(names, sorted(names))

    def test_no_duplicates(self):
        u"""Les doublons (meme nom) sont elimines."""
        html = (
            '<a href="coord/foo.dat">foo.dat</a> \\ A<br>'
            '<a href="coord/foo.dat">foo.dat</a> \\ B<br>')
        entries = parse_index_html(html)
        self.assertEqual(len(entries), 1)

    def test_empty_html(self):
        u"""HTML sans entree retourne liste vide."""
        self.assertEqual(parse_index_html(''), [])
        self.assertEqual(parse_index_html('<html></html>'), [])

    def test_distinguishes_coord_paths(self):
        u"""coord/ et coord_updates/ produisent des URLs differentes."""
        entries = parse_index_html(_HTML_SAMPLE)
        d = {e.name: e.dat_url for e in entries}
        self.assertIn('/coord/', d['naca2412.dat'])
        self.assertIn('/coord_updates/', d['bacnlf.dat'])


class TestUIUCLoaderCache(unittest.TestCase):
    u"""Tests du systeme de cache."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp(prefix='uiuc_test_')
        self.loader = UIUCLoader(cache_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_default_cache_dir(self):
        u"""Cache par defaut sous le home utilisateur."""
        from model.uiuc_loader import default_cache_dir
        d = default_cache_dir()
        self.assertIn('.airfoiltools', str(d))
        self.assertTrue(str(d).endswith(os.path.join('uiuc')))

    def test_index_cache_written(self):
        u"""fetch_index ecrit le cache JSON."""
        with mock.patch.object(
                UIUCLoader, '_http_get_text',
                return_value=_HTML_SAMPLE):
            self.loader.fetch_index()
        cache_file = Path(self.tmpdir) / 'index.json'
        self.assertTrue(cache_file.exists())
        with open(cache_file) as f:
            data = json.load(f)
        self.assertIn('timestamp', data)
        self.assertIn('entries', data)
        self.assertGreater(len(data['entries']), 0)

    def test_index_cache_used(self):
        u"""Deuxieme appel utilise le cache (pas d'HTTP)."""
        with mock.patch.object(
                UIUCLoader, '_http_get_text',
                return_value=_HTML_SAMPLE) as m:
            self.loader.fetch_index()
            self.loader.fetch_index()
            self.assertEqual(m.call_count, 1)

    def test_force_bypasses_cache(self):
        u"""force=True ignore le cache."""
        with mock.patch.object(
                UIUCLoader, '_http_get_text',
                return_value=_HTML_SAMPLE) as m:
            self.loader.fetch_index()
            self.loader.fetch_index(force=True)
            self.assertEqual(m.call_count, 2)

    def test_expired_cache_refetched(self):
        u"""Cache expire (> TTL) est ignore."""
        cache_file = Path(self.tmpdir) / 'index.json'
        old_data = {
            'timestamp': time.time() - INDEX_TTL_SECONDS - 100,
            'entries': [{
                'name': 'old.dat',
                'description': 'old',
                'dat_url': 'http://x'}],
        }
        with open(cache_file, 'w') as f:
            json.dump(old_data, f)
        with mock.patch.object(
                UIUCLoader, '_http_get_text',
                return_value=_HTML_SAMPLE) as m:
            entries = self.loader.fetch_index()
            self.assertEqual(m.call_count, 1)
            self.assertNotIn(
                'old.dat', [e.name for e in entries])

    def test_dat_cache_written(self):
        u"""fetch_dat ecrit le fichier dans le cache."""
        entry = UIUCEntry(
            name='foo.dat', description='',
            dat_url='http://example.com/foo.dat')
        with mock.patch.object(
                UIUCLoader, '_http_get_bytes',
                return_value=b'1.0 0.0\n0.5 0.05\n'):
            path = self.loader.fetch_dat(entry)
        self.assertTrue(path.exists())
        self.assertEqual(path.read_bytes(),
                         b'1.0 0.0\n0.5 0.05\n')

    def test_dat_cache_used(self):
        u"""fetch_dat utilise le cache si fichier deja present."""
        entry = UIUCEntry(
            name='bar.dat', description='',
            dat_url='http://example.com/bar.dat')
        with mock.patch.object(
                UIUCLoader, '_http_get_bytes',
                return_value=b'data') as m:
            self.loader.fetch_dat(entry)
            self.loader.fetch_dat(entry)
            self.assertEqual(m.call_count, 1)


if __name__ == '__main__':
    unittest.main()
