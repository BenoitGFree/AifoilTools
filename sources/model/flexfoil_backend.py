#!/usr/bin/python
# -*- coding: utf-8 -*-

u"""
Backend FlexFoil : solveur 2D alternatif a XFoil.exe.

FlexFoil (https://foil.flexcompute.com, licence MIT) est une
reimplementation en Rust du solveur XFoil de Mark Drela, appelee comme
bibliotheque Python native (pip install flexfoil ; wheels cp311). Ce
module l'integre derriere la meme abstraction Preprocessor / Simulator /
Postprocessor que XFoil, en produisant le MEME contrat de donnees
(voir base.AbstractPostprocessor), si bien que la GUI et les objets
SimulationResults fonctionnent sans modification.

Particularites de FlexFoil exploitees ici :

- Le solveur visqueux n'expose PAS de Cp visqueux. On le reconstruit a
  partir de la vitesse de bord de couche limite : Cp = 1 - (Ue/Vinf)^2
  (relation incompressible, exacte a Mach 0 ; validee a 2e-3 pres contre
  le CPWR visqueux de XFoil).
- La couche limite est fournie deja separee extrados / intrados, mais
  sans l'ordonnee y de surface. On reconstruit y par interpolation sur
  la geometrie panelisee, puis on assemble un contour continu facon
  Selig (BF -> extrados -> BA -> intrados -> BF) avec une abscisse
  curviligne s, pour rester compatible avec le contrat XFoil.
- Le Cp non visqueux provient de la passe non visqueuse native (un Cp
  par noeud de panneau).

@author: Nervures
@date: 2026-06
"""

import os
import pickle
import logging

import numpy as np

from .base import AbstractPreprocessor, AbstractSimulator, AbstractPostprocessor
from .foilconfig import load_defaults, merge_params

logger = logging.getLogger(__name__)

# Noms des fichiers de handoff dans le repertoire de travail
_CASE_FILE = 'flexfoil_case.pkl'      # pre -> sim : geometrie + parametres
_RAW_FILE = 'flexfoil_raw.pkl'        # sim -> post : resultats natifs bruts


def is_available():
    u"""Indique si la bibliotheque flexfoil est importable.

    :rtype: bool
    """
    try:
        import flexfoil  # noqa: F401
        return True
    except Exception:
        return False


def _re_list(params):
    u"""Liste des Reynolds depuis les parametres (RE_LIST ou RE)."""
    re_list = params.get('RE_LIST', [params.get('RE', 1e6)])
    if not isinstance(re_list, (list, tuple)):
        re_list = [re_list]
    return [float(r) for r in re_list]


def _alpha_list(params):
    u"""Liste des incidences depuis ALPHA_MIN / ALPHA_MAX / ALPHA_STEP."""
    a0 = float(params.get('ALPHA_MIN', -5.0))
    a1 = float(params.get('ALPHA_MAX', 15.0))
    da = float(params.get('ALPHA_STEP', 0.5))
    if da <= 0:
        da = 0.5
    alphas = []
    a = a0
    while a <= a1 + 1e-9:
        alphas.append(round(a, 6))
        a += da
    return alphas


def _flat(coords):
    u"""Convertit une liste/array de points (n,2) en liste plate [x,y,...]."""
    arr = np.asarray(coords, dtype=float).reshape(-1, 2)
    flat = []
    for x, y in arr:
        flat.append(float(x))
        flat.append(float(y))
    return flat


# ======================================================================
#  Preprocessor
# ======================================================================

class FlexFoilPreprocessor(AbstractPreprocessor):
    u"""Prepare la geometrie et les parametres pour FlexFoil.

    N'ecrit pas de fichier de commandes (FlexFoil est une bibliotheque) :
    serialise la geometrie panelisee et les parametres resolus dans un
    unique fichier pickle, consomme par le simulateur.
    """

    def prepare(self, profile_points, params=None):
        u"""Resout les parametres, panelise le profil, serialise le cas.

        :param profile_points: coordonnees normalisees [0,1], shape (n, 2),
            ordre Selig
        :param params: parametres utilisateur (surchargent les defauts)
        :returns: [chemin du fichier de cas]
        :rtype: list[str]
        """
        defaults = load_defaults('flexfoil')
        p = merge_params(defaults, params)

        if not os.path.isdir(self.work_dir):
            os.makedirs(self.work_dir)

        pts = np.asarray(profile_points, dtype=float).reshape(-1, 2)

        # Repaneling FlexFoil (algorithme XFoil base courbure), sinon brut
        coords = pts
        if p.get('REPANEL', True):
            try:
                from flexfoil import _rustfoil as rf
                npanel = int(p.get('NPANEL', 200))
                paneled = rf.repanel_xfoil(_flat(pts), npanel)
                coords = np.asarray(paneled, dtype=float)
            except Exception as e:
                logger.warning(u"Repaneling FlexFoil indisponible (%s) : "
                               u"geometrie brute utilisee", e)
                coords = pts

        case = {'coords': coords, 'params': p}
        path = os.path.join(self.work_dir, _CASE_FILE)
        with open(path, 'wb') as f:
            pickle.dump(case, f, protocol=2)
        return [path]


# ======================================================================
#  Simulator
# ======================================================================

class FlexFoilSimulator(AbstractSimulator):
    u"""Execute FlexFoil en bibliotheque native (pas d'executable).

    Pour chaque Reynolds : balayage visqueux (parallelise) + couche
    limite par incidence. Une passe non visqueuse (independante du
    Reynolds) fournit le Cp non visqueux. Les resultats natifs bruts sont
    serialises pour le postprocesseur.
    """

    def __init__(self, exe_path=None, timeout=60):
        # FlexFoil n'a pas d'executable : exe_path est ignore (garde pour
        # l'uniformite de la signature avec les autres simulateurs).
        super(FlexFoilSimulator, self).__init__(exe_path or 'flexfoil',
                                                timeout)

    def run(self, work_dir, input_files):
        u"""Lance les calculs FlexFoil et serialise les resultats bruts.

        :param work_dir: repertoire de travail
        :param input_files: [fichier de cas] genere par le preprocesseur
        :returns: True si au moins un point a converge
        :rtype: bool
        """
        try:
            from flexfoil import _rustfoil as rf
        except Exception as e:
            logger.error(u"FlexFoil indisponible : %s", e)
            return False

        case_path = os.path.join(work_dir, _CASE_FILE)
        if not os.path.isfile(case_path):
            logger.error(u"Fichier de cas FlexFoil introuvable : %s",
                         case_path)
            return False
        with open(case_path, 'rb') as f:
            case = pickle.load(f)

        coords = np.asarray(case['coords'], dtype=float)
        p = case['params']
        flat = _flat(coords)

        re_list = _re_list(p)
        alphas = _alpha_list(p)
        mach = float(p.get('MACH', 0.0))
        ncrit = float(p.get('NCRIT', 9.0))
        max_iter = int(p.get('ITER', 100))
        xtr_u = float(p.get('XTR_TOP', 1.0))
        xtr_l = float(p.get('XTR_BOT', 1.0))
        viscous = bool(p.get('VISCOUS', True))

        raw = {
            'panel_xy': coords.tolist(),
            'visc': {},        # {re: {alpha: dict natif analyze_faithful}}
            'bl': {},          # {re: {alpha: dict natif get_bl_distribution}}
            'inviscid': {},    # {alpha: dict natif analyze_inviscid}
        }

        any_conv = False

        # --- Passes visqueuses par Reynolds ---
        if viscous:
            for re_val in re_list:
                raw['visc'][re_val] = {}
                raw['bl'][re_val] = {}
                # Balayage parallelise (cl/cd/cm/transition par incidence)
                try:
                    batch = rf.analyze_faithful_batch(
                        flat, alphas, re_val, mach, ncrit, max_iter,
                        1, xtr_u, xtr_l)
                except Exception as e:
                    logger.warning(u"analyze_faithful_batch echec "
                                   u"(Re=%g) : %s", re_val, e)
                    batch = []
                for a, d in zip(alphas, batch):
                    raw['visc'][re_val][a] = d
                    if d.get('converged'):
                        any_conv = True
                # Couche limite par incidence (pas de version batch native)
                for a in alphas:
                    d = raw['visc'][re_val].get(a, {})
                    if not d.get('converged'):
                        continue
                    try:
                        bl = rf.get_bl_distribution(
                            flat, a, re_val, mach, ncrit, max_iter,
                            1, xtr_u, xtr_l)
                    except Exception as e:
                        logger.warning(u"get_bl_distribution echec "
                                       u"(Re=%g a=%g) : %s", re_val, a, e)
                        continue
                    if bl.get('success', True):
                        raw['bl'][re_val][a] = bl
        else:
            # Mode non visqueux : la polaire vient de la passe panneaux
            re_list = []  # pas de Reynolds pour les polaires visqueuses

        # --- Passe non visqueuse (Cp non visqueux, par alpha) ---
        if p.get('INVISCID_CP', True) or not viscous:
            try:
                inv = rf.analyze_inviscid_batch(flat, alphas)
                for a, d in zip(alphas, inv):
                    raw['inviscid'][a] = d
                    if not viscous and d.get('success'):
                        any_conv = True
            except Exception as e:
                logger.warning(u"analyze_inviscid_batch echec : %s", e)

        raw_path = os.path.join(work_dir, _RAW_FILE)
        with open(raw_path, 'wb') as f:
            pickle.dump(raw, f, protocol=2)

        return any_conv


# ======================================================================
#  Postprocessor
# ======================================================================

class FlexFoilPostprocessor(AbstractPostprocessor):
    u"""Assemble les resultats natifs FlexFoil dans le contrat commun.

    Produit polars / cp / cpi / bl / warnings au format attendu par
    SimulationResults (cf. AbstractPostprocessor).
    """

    def parse(self, work_dir):
        results = {'polars': {}, 'cp': {}, 'cpi': {}, 'bl': {},
                   'warnings': []}

        raw_path = os.path.join(work_dir, _RAW_FILE)
        if not os.path.isfile(raw_path):
            results['warnings'].append(
                u"Resultats FlexFoil introuvables : %s" % raw_path)
            return results
        with open(raw_path, 'rb') as f:
            raw = pickle.load(f)

        panel_xy = np.asarray(raw.get('panel_xy', []), dtype=float)
        if panel_xy.ndim != 2 or panel_xy.shape[0] < 4:
            results['warnings'].append(u"Geometrie panelisee invalide.")
            panel_xy = None

        # --- Polaires (par Reynolds) ---
        for re_val, alpha_dict in raw.get('visc', {}).items():
            rows = []
            for a in sorted(alpha_dict.keys()):
                d = alpha_dict[a]
                if not d.get('converged'):
                    continue
                rows.append([
                    float(a),
                    float(d.get('cl', 0.0)),
                    float(d.get('cd', 0.0)),
                    float(d.get('cd_pressure', 0.0)),
                    float(d.get('cm', 0.0)),
                    float(d.get('x_tr_upper', 1.0)),
                    float(d.get('x_tr_lower', 1.0)),
                ])
            if not rows:
                continue
            arr = np.array(rows)
            results['polars'][float(re_val)] = {
                'alpha': arr[:, 0], 'CL': arr[:, 1], 'CD': arr[:, 2],
                'CDp': arr[:, 3], 'CM': arr[:, 4],
                'Top_Xtr': arr[:, 5], 'Bot_Xtr': arr[:, 6],
            }

        # --- Cp visqueux (reconstruit 1 - Ue^2) + couche limite ---
        for re_val, alpha_dict in raw.get('bl', {}).items():
            re_f = float(re_val)
            for a, bl in alpha_dict.items():
                surf = self._surfaces(bl, panel_xy)
                if surf is None:
                    continue
                (Xe, Ye, ue_e, dse, the, cfe, he), \
                    (Xi, Yi, ue_i, dsi, thi, cfi, hi) = surf

                # Cp = 1 - Ue^2 (Selig : extrados TE->BA puis intrados)
                cp_e = 1.0 - ue_e ** 2
                cp_i = 1.0 - ue_i ** 2
                X = np.concatenate([Xe, Xi])
                Y = np.concatenate([Ye, Yi])
                Cp = np.concatenate([cp_e, cp_i])
                results['cp'].setdefault(re_f, {})[float(a)] = \
                    np.column_stack([X, Y, Cp])

                # Couche limite : contour continu + abscisse curviligne s
                Ue = np.concatenate([ue_e, ue_i])
                Ds = np.concatenate([dse, dsi])
                Th = np.concatenate([the, thi])
                Cf = np.concatenate([cfe, cfi])
                H = np.concatenate([he, hi])
                s = self._arclength(X, Y)
                results['bl'].setdefault(re_f, {})[float(a)] = {
                    's': s, 'x': X, 'y': Y, 'Ue_Vinf': Ue,
                    'Dstar': Ds, 'Theta': Th, 'Cf': Cf, 'H': H,
                }

        # --- Cp non visqueux (par alpha, independant du Reynolds) ---
        for a, d in raw.get('inviscid', {}).items():
            if not d.get('success'):
                continue
            cp = np.asarray(d.get('cp', []), dtype=float)
            if cp.size == 0:
                continue
            if panel_xy is not None and len(cp) == panel_xy.shape[0]:
                # Cp par noeud de panneau : appariement direct (x, y, Cp)
                arr = np.column_stack([panel_xy[:, 0], panel_xy[:, 1], cp])
            else:
                cp_x = np.asarray(d.get('cp_x', []), dtype=float)
                n = min(len(cp), len(cp_x))
                if n == 0:
                    continue
                arr = np.column_stack([cp_x[:n], cp[:n]])
            results['cpi'][float(a)] = arr

        n_pol = len(results['polars'])
        n_cp = sum(len(v) for v in results['cp'].values())
        n_bl = sum(len(v) for v in results['bl'].values())
        n_cpi = len(results['cpi'])
        logger.info(u"FlexFoil : %d polaires, %d Cp, %d Cp non visqueux, "
                    u"%d couches limites", n_pol, n_cp, n_cpi, n_bl)

        return results

    # ------------------------------------------------------------------
    #  Helpers d'assemblage
    # ------------------------------------------------------------------

    @staticmethod
    def _arclength(x, y):
        u"""Abscisse curviligne cumulee le long du contour (x, y)."""
        d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        return np.concatenate([[0.0], np.cumsum(d)])

    @staticmethod
    def _surfaces(bl, panel_xy):
        u"""Construit les arcs extrados / intrados ordonnes Selig.

        Reconstruit l'ordonnee y de chaque station de couche limite par
        interpolation sur la geometrie panelisee (FlexFoil ne fournit pas
        y), puis ordonne :
        - extrados : x decroissant (BF -> BA) ;
        - intrados : x croissant (BA -> BF).

        :returns: ((Xe,Ye,Ue_e,Ds_e,Th_e,Cf_e,H_e),
                   (Xi,Yi,Ue_i,Ds_i,Th_i,Cf_i,H_i)) ou None
        """
        try:
            xu = np.asarray(bl['x_upper'], dtype=float)
            xl = np.asarray(bl['x_lower'], dtype=float)
        except (KeyError, TypeError):
            return None
        if xu.size < 2 or xl.size < 2 or panel_xy is None:
            return None

        px, py = panel_xy[:, 0], panel_xy[:, 1]
        le = int(np.argmin(px))
        # Tables d'interpolation y(x), x croissant, pour chaque surface
        up_x = px[:le + 1][::-1]
        up_y = py[:le + 1][::-1]
        lo_x = px[le:]
        lo_y = py[le:]
        # Garantir la monotonie stricte des tables (interp l'exige)
        up_x, up_y = FlexFoilPostprocessor._dedup(up_x, up_y)
        lo_x, lo_y = FlexFoilPostprocessor._dedup(lo_x, lo_y)

        # Reconstruction de y avec gestion de l'enroulement au point d'arret.
        # A incidence non nulle, l'arret quitte le BA : la couche limite d'un
        # cote contourne le nez (ses 1eres stations, en ordre naturel, vont
        # de l'arret jusqu'au BA en DECROISSANT en x, donc sur le nez de
        # l'AUTRE surface). Interpoler tout le cote sur sa seule table donne
        # un y errone pour cette portion d'enroulement -> saut vertical au BA.
        # On bascule donc de table au point de rebroussement (x minimal de la
        # station, = le BA dans l'ordre naturel) : avant lui, on lit le nez
        # oppose ; apres, la surface principale.
        def _reconstruct_y(xs, main_up):
            k = int(np.argmin(xs))         # rebroussement = BA
            y = np.empty_like(xs)
            main_x, main_y = (up_x, up_y) if main_up else (lo_x, lo_y)
            opp_x, opp_y = (lo_x, lo_y) if main_up else (up_x, up_y)
            if k > 0:                       # portion d'enroulement (nez oppose)
                y[:k] = np.interp(xs[:k], opp_x, opp_y)
            y[k:] = np.interp(xs[k:], main_x, main_y)
            return y

        yu = _reconstruct_y(xu, main_up=True)
        yl = _reconstruct_y(xl, main_up=False)

        def _get(key_u, key_l):
            return (np.asarray(bl.get(key_u, []), dtype=float),
                    np.asarray(bl.get(key_l, []), dtype=float))

        ueu, uel = _get('ue_upper', 'ue_lower')
        dsu, dsl = _get('delta_star_upper', 'delta_star_lower')
        thu, thl = _get('theta_upper', 'theta_lower')
        cfu, cfl = _get('cf_upper', 'cf_lower')
        hu, hl = _get('h_upper', 'h_lower')

        # Ordre des stations : on PRESERVE l'ordre naturel de FlexFoil
        # (stations integrees par abscisse curviligne depuis le point
        # d'arret, du BA vers le BF). NE PAS trier par x : des que le point
        # d'arret quitte le BA (toute incidence non nulle / profil cambre),
        # la surface portant l'arret est double-valuee en x pres du BA
        # (la couche limite contourne le nez). Un argsort(x) entrelace alors
        # les deux branches et produit des zigzags (Cp alternant succion /
        # pression). L'ordre naturel reste un contour continu.
        # Extrados : sens inverse (BF -> BA, x globalement decroissant) ;
        # intrados : sens naturel (BA -> BF, x globalement croissant).
        oe = np.arange(xu.size)[::-1]
        oi = np.arange(xl.size)

        def _pack(order, X, Y, Ue, Ds, Th, Cf, H):
            return (X[order], Y[order], Ue[order], Ds[order],
                    Th[order], Cf[order], H[order])

        extr = _pack(oe, xu, yu, ueu, dsu, thu, cfu, hu)
        intr = _pack(oi, xl, yl, uel, dsl, thl, cfl, hl)
        return extr, intr

    @staticmethod
    def _dedup(x, y):
        u"""Rend x strictement croissant pour np.interp (retire doublons)."""
        keep = np.concatenate([[True], np.diff(x) > 1e-12])
        return x[keep], y[keep]
