#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Tests du package foil2d.

Test 1 : Config - lecture et fusion des parametres
Test 2 : Preprocessor - generation des fichiers profil et commandes
Test 3 : Postprocessor - parsing de fichiers de sortie XFoil synthetiques
Test 4 : Pipeline - enchainement prepare_only + parse_only

Ces tests ne necessitent PAS l'executable XFoil.

Usage::

    python sources/model/aerodynamique/foil2d/test_foil2d.py

@author: Nervures
@date: 2026-02
"""

import os
import sys
import tempfile
import shutil
import numpy as np

# Ajouter le repertoire sources/ au path
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.normpath(os.path.join(_here, '..', 'sources'))
if _src not in sys.path:
    sys.path.insert(0, _src)

from model.foilconfig import load_defaults, merge_params, load_config
from model.xfoil_preprocessor import XFoilPreprocessor
from model.xfoil_postprocessor import XFoilPostprocessor
from model.pipeline import FoilAnalysisPipeline


def make_naca0012(n=50):
    u"""Genere un profil NACA 0012 simplifie pour les tests.

    :param n: nombre de points par surface
    :type n: int
    :returns: coordonnees (2*n+1, 2)
    :rtype: numpy.ndarray
    """
    t = 0.12  # epaisseur relative
    x = np.linspace(0, 1, n)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1015 * x**4)
    # Extrados (BF -> BA) puis intrados (BA -> BF)
    x_ext = x[::-1]
    y_ext = yt[::-1]
    x_int = x[1:]  # eviter le doublon au BA
    y_int = -yt[1:]
    x_all = np.concatenate([x_ext, x_int])
    y_all = np.concatenate([y_ext, y_int])
    return np.column_stack([x_all, y_all])


def write_fake_polar(work_dir, re_val):
    u"""Ecrit un fichier polaire XFoil synthetique pour tester le parsing.

    :param work_dir: repertoire de sortie
    :type work_dir: str
    :param re_val: valeur de Reynolds
    :type re_val: float
    """
    filepath = os.path.join(work_dir, 'polar_Re%g.dat' % re_val)
    with open(filepath, 'w') as f:
        f.write(' XFOIL         Version 6.99\n')
        f.write('\n')
        f.write(' Calculated polar for: Profil Axile\n')
        f.write('\n')
        f.write(' 1 1 Reynolds number fixed          Mach number fixed\n')
        f.write('\n')
        f.write(' xtrf =   0.010 (top)        0.010 (bottom)\n')
        f.write(' Mach =   0.000     Re =     0.500 e 6     Ncrit =   9.000\n')
        f.write('\n')
        f.write('  alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr\n')
        f.write(' ------ -------- --------- --------- -------- -------- --------\n')
        # Donnees synthetiques
        for alpha in np.arange(-2, 10.5, 0.5):
            cl = 0.11 * alpha
            cd = 0.008 + 0.001 * alpha**2
            cdp = cd * 0.6
            cm = -0.02
            xtr_top = max(0.01, 0.5 - 0.03 * alpha)
            xtr_bot = min(1.0, 0.5 + 0.05 * alpha)
            f.write('  %6.2f  %7.4f   %7.5f   %7.5f  %7.4f  %6.4f   %6.4f\n'
                    % (alpha, cl, cd, cdp, cm, xtr_top, xtr_bot))


def write_fake_cp(work_dir, re_val, alpha):
    u"""Ecrit un fichier Cp synthetique.

    :param work_dir: repertoire de sortie
    :type work_dir: str
    :param re_val: valeur de Reynolds
    :type re_val: float
    :param alpha: angle d'attaque
    :type alpha: float
    """
    filepath = os.path.join(work_dir, 'cp_Re%g_a%g.dat' % (re_val, alpha))
    with open(filepath, 'w') as f:
        f.write('#  x        Cp\n')
        for x in np.linspace(0, 1, 20):
            cp = 1.0 - 4.0 * x * (1.0 - x)  # parabole inversee
            f.write(' %8.5f  %8.5f\n' % (x, cp))


# =========================================================================
# Tests
# =========================================================================

def test_config():
    u"""Test de la lecture de configuration."""
    print("--- Test config ---")
    defaults = load_defaults('xfoil')
    assert 'RE' in defaults, "RE manquant dans defaults"
    assert 'NCRIT' in defaults, "NCRIT manquant dans defaults"
    assert defaults['VISCOUS'] is True, "VISCOUS devrait etre True"
    assert isinstance(defaults['RE'], (int, float)), "RE devrait etre numerique"
    assert defaults['TIMEOUT'] == 30, "TIMEOUT devrait etre 30"

    # Test merge
    user = {'RE': 1000000, 'NCRIT': 5}
    merged = merge_params(defaults, user)
    assert merged['RE'] == 1000000, "RE devrait etre surcharge"
    assert merged['NCRIT'] == 5, "NCRIT devrait etre surcharge"
    assert merged['MACH'] == defaults['MACH'], "MACH ne devrait pas changer"
    print("  OK : config chargee, %d parametres" % len(defaults))
    print("  OK : merge params fonctionne")


def test_preprocessor():
    u"""Test de la generation des fichiers d'entree."""
    print("--- Test preprocessor ---")
    work_dir = tempfile.mkdtemp(prefix='test_foil2d_pre_')
    try:
        pre = XFoilPreprocessor(work_dir)
        points = make_naca0012()
        params = {
            'RE': 500000,
            'ALPHA_MIN': -2.0,
            'ALPHA_MAX': 10.0,
            'ALPHA_STEP': 1.0
        }
        files = pre.prepare(points, params)

        # Verifier que les fichiers sont crees
        assert len(files) >= 2, "Au moins 2 fichiers attendus (profil + cmd)"
        for f in files:
            assert os.path.isfile(f), "Fichier manquant : %s" % f

        # Verifier le contenu du profil
        profil_file = os.path.join(work_dir, 'profil.dat')
        assert os.path.isfile(profil_file), "profil.dat manquant"
        with open(profil_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == points.shape[0] + 1, \
            "Nombre de lignes incorrect dans profil.dat"

        # Verifier le contenu du script de commandes
        cmd_file = os.path.join(work_dir, 'xfoil_alpha.cmd')
        assert os.path.isfile(cmd_file), "xfoil_alpha.cmd manquant"
        with open(cmd_file, 'r') as f:
            content = f.read()
        assert 'LOAD profil.dat' in content, "LOAD manquant"
        assert 'VISC' in content, "VISC manquant"
        assert 'ASEQ' in content, "ASEQ manquant"
        assert 'QUIT' in content, "QUIT manquant"

        print("  OK : %d fichiers generes" % len(files))
        print("  OK : profil.dat contient %d points" % points.shape[0])
        print("  OK : xfoil_alpha.cmd contient les commandes attendues")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_preprocessor_multi_re():
    u"""Test du preprocessor avec plusieurs Reynolds."""
    print("--- Test preprocessor multi-Re ---")
    work_dir = tempfile.mkdtemp(prefix='test_foil2d_mre_')
    try:
        pre = XFoilPreprocessor(work_dir)
        points = make_naca0012()
        params = {
            'RE_LIST': [300000, 500000, 1000000],
            'ALPHA_MIN': 0.0,
            'ALPHA_MAX': 5.0,
            'ALPHA_STEP': 1.0
        }
        files = pre.prepare(points, params)

        cmd_file = os.path.join(work_dir, 'xfoil_alpha.cmd')
        with open(cmd_file, 'r') as f:
            content = f.read()

        # 3 Re => 1er VISC Re + 2x (VISC toggle OFF + VISC Re) = 5 VISC
        assert content.count('VISC') == 5, \
            "5 VISC attendus (toggle OFF/ON), trouve %d" % content.count('VISC')
        assert 'VISC 300000' in content
        assert 'VISC 500000' in content
        assert 'VISC 1e+06' in content or 'VISC 1000000' in content
        # Verifier les 3 fichiers polaire avec nommage entier
        assert 'polar_Re300000.dat' in content
        assert 'polar_Re500000.dat' in content
        assert 'polar_Re1000000.dat' in content

        print("  OK : 3 blocs Reynolds generes")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_postprocessor():
    u"""Test du parsing des fichiers de sortie."""
    print("--- Test postprocessor ---")
    work_dir = tempfile.mkdtemp(prefix='test_foil2d_post_')
    try:
        # Creer des fichiers synthetiques
        write_fake_polar(work_dir, 500000)
        write_fake_cp(work_dir, 500000, 5.0)

        post = XFoilPostprocessor()
        results = post.parse(work_dir)

        # Verifier la polaire
        assert 500000.0 in results['polars'], \
            "Polaire Re=500000 manquante"
        polar = results['polars'][500000.0]
        assert 'alpha' in polar, "alpha manquant"
        assert 'CL' in polar, "CL manquant"
        assert 'CD' in polar, "CD manquant"
        assert len(polar['alpha']) == 25, \
            "25 points attendus, trouve %d" % len(polar['alpha'])

        # Verifier les Cp
        assert 500000.0 in results['cp'], "Cp Re=500000 manquant"
        assert 5.0 in results['cp'][500000.0], "Cp alpha=5.0 manquant"
        cp_data = results['cp'][500000.0][5.0]
        assert cp_data.shape == (20, 2), \
            "Shape Cp incorrect : %s" % str(cp_data.shape)

        print("  OK : polaire parsee, %d points" % len(polar['alpha']))
        print("  OK : CL range [%.3f, %.3f]"
              % (polar['CL'].min(), polar['CL'].max()))
        print("  OK : Cp parse, shape %s" % str(cp_data.shape))
        print("  OK : %d warnings" % len(results['warnings']))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pipeline_prepare_and_parse():
    u"""Test du pipeline (prepare + parse, sans simulation)."""
    print("--- Test pipeline (prepare + parse) ---")
    work_dir = tempfile.mkdtemp(prefix='test_foil2d_pipe_')
    try:
        pipeline = FoilAnalysisPipeline(solver='xfoil', work_dir=work_dir)
        points = make_naca0012()

        # Etape 1 : prepare_only
        files = pipeline.prepare_only(points, {
            'RE': 500000,
            'ALPHA_MIN': -2.0,
            'ALPHA_MAX': 10.0,
            'ALPHA_STEP': 0.5
        })
        assert len(files) >= 2

        # Simuler des resultats (sans XFoil)
        write_fake_polar(work_dir, 500000)
        write_fake_cp(work_dir, 500000, 5.0)

        # Etape 3 : parse_only
        results = pipeline.parse_only()
        assert 500000.0 in results['polars']
        assert len(results['polars'][500000.0]['alpha']) > 0

        print("  OK : pipeline prepare_only + parse_only fonctionne")
        print("  OK : %d polaires, %d distributions Cp"
              % (len(results['polars']),
                 sum(len(v) for v in results['cp'].values())))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Tests foil2d")
    print("=" * 60)
    try:
        test_config()
        test_preprocessor()
        test_preprocessor_multi_re()
        test_postprocessor()
        test_pipeline_prepare_and_parse()
        print("=" * 60)
        print("TOUS LES TESTS OK")
        print("=" * 60)
    except AssertionError as e:
        print("ECHEC : %s" % str(e))
        sys.exit(1)
    except Exception as e:
        print("ERREUR : %s" % str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
