#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Objets metier pour la simulation aerodynamique 2D.

- SimulationResults : resultats structures avec properties calculees
  et methodes de trace composables (comparaison multi-simulations).
- Simulation : lie un Profil a des parametres et orchestre le pipeline.

Usage::

    from model.aerodynamique.foil2d.simulation import Simulation
    from model.aerodynamique.foil2d.profil import Profil

    profil = Profil.from_naca('2412')
    sim = Simulation(profil, params={'RE': 500000})
    results = sim.run()
    results.plot_polars()

Comparaison entre simulations::

    ax = sim1.results.plot_cl(label='NACA 2412', show=False)
    sim2.results.plot_cl(ax=ax, label='NACA 0012')

@author: Nervures
@date: 2026-02
"""

import os
import copy
import tempfile
import logging

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
#  SimulationResults
# ======================================================================

class SimulationResults(object):
    u"""Resultats structures d'une simulation aerodynamique 2D.

    Encapsule les donnees brutes retournees par le pipeline
    (polaires, Cp, couche limite) et fournit :
    - des properties calculees (cl_max, finesse_max, ...)
    - des methodes de trace composables pour la comparaison
    """

    def __init__(self, polars=None, cp=None, bl=None, warnings=None):
        u"""
        :param polars: polaires par Reynolds
        :type polars: dict or None
        :param cp: distributions Cp par (Re, alpha)
        :type cp: dict or None
        :param bl: donnees couche limite par Re
        :type bl: dict or None
        :param warnings: messages d'avertissement
        :type warnings: list or None
        """
        self._polars = polars if polars is not None else {}
        self._cp = cp if cp is not None else {}
        self._bl = bl if bl is not None else {}
        self._warnings = warnings if warnings is not None else []

    @classmethod
    def from_dict(cls, results_dict):
        u"""Cree une instance depuis le dict retourne par pipeline.run().

        :param results_dict: resultats bruts du pipeline
        :type results_dict: dict
        :returns: instance SimulationResults
        :rtype: SimulationResults
        """
        return cls(
            polars=copy.deepcopy(results_dict.get('polars', {})),
            cp=copy.deepcopy(results_dict.get('cp', {})),
            bl=copy.deepcopy(results_dict.get('bl', {})),
            warnings=list(results_dict.get('warnings', []))
        )

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def polars(self):
        u"""Polaires brutes : dict {Re: {'alpha': ndarray, 'CL': ndarray, ...}}."""
        return self._polars

    @property
    def cp(self):
        u"""Distributions Cp : dict {Re: {alpha: ndarray(n,2)}}."""
        return self._cp

    @property
    def bl(self):
        u"""Donnees couche limite : dict {Re: dict or None}."""
        return self._bl

    @property
    def warnings(self):
        u"""Messages d'avertissement."""
        return self._warnings

    @property
    def re_list(self):
        u"""Liste triee des Reynolds pour lesquels on a des polaires.

        :rtype: list[float]
        """
        return sorted(re for re in self._polars if isinstance(re, float))

    @property
    def has_polars(self):
        u"""True si au moins une polaire disponible."""
        return len(self._polars) > 0

    @property
    def has_cp(self):
        u"""True si au moins une distribution Cp disponible."""
        return any(len(alphas) > 0 for alphas in self._cp.values())

    @property
    def n_converged(self):
        u"""Nombre total de points alpha converges (toutes polaires).

        :rtype: int
        """
        n = 0
        for re_val in self._polars:
            polar = self._polars[re_val]
            if isinstance(polar, dict) and 'alpha' in polar:
                n += len(polar['alpha'])
        return n

    # ------------------------------------------------------------------
    #  Methodes d'acces
    # ------------------------------------------------------------------

    def _resolve_re(self, re):
        u"""Resout le Re : si None, retourne le premier de re_list.

        :param re: Reynolds demande ou None
        :type re: float or None
        :returns: Re resolu
        :rtype: float or None
        """
        if re is not None:
            return float(re)
        re_vals = self.re_list
        if re_vals:
            return re_vals[0]
        return None

    def get_polar(self, re=None):
        u"""Retourne la polaire pour un Re donne.

        :param re: Reynolds (None = premier disponible)
        :type re: float or None
        :returns: dict avec 'alpha', 'CL', 'CD', ... ou None
        :rtype: dict or None
        """
        re_val = self._resolve_re(re)
        if re_val is None:
            return None
        return self._polars.get(re_val, None)

    def get_cp(self, re, alpha):
        u"""Retourne la distribution Cp pour (Re, alpha).

        :param re: Reynolds
        :type re: float
        :param alpha: angle d'attaque
        :type alpha: float
        :returns: ndarray(n, 2) [x, Cp] ou None
        :rtype: numpy.ndarray or None
        """
        re_val = float(re)
        if re_val not in self._cp:
            return None
        return self._cp[re_val].get(alpha, None)

    def alpha_range(self, re=None):
        u"""Retourne (alpha_min, alpha_max) pour un Re donne.

        :param re: Reynolds (None = premier disponible)
        :type re: float or None
        :returns: (alpha_min, alpha_max) ou None
        :rtype: tuple or None
        """
        polar = self.get_polar(re)
        if polar is None or 'alpha' not in polar:
            return None
        alpha = polar['alpha']
        if len(alpha) == 0:
            return None
        return (float(alpha.min()), float(alpha.max()))

    def cl_max(self, re=None):
        u"""Retourne (CL_max, alpha_CL_max) pour un Re donne.

        :param re: Reynolds (None = premier disponible)
        :type re: float or None
        :returns: (CL_max, alpha_CL_max) ou None
        :rtype: tuple or None
        """
        polar = self.get_polar(re)
        if polar is None or 'CL' not in polar:
            return None
        cl = polar['CL']
        alpha = polar['alpha']
        if len(cl) == 0:
            return None
        idx = np.argmax(cl)
        return (float(cl[idx]), float(alpha[idx]))

    def finesse_max(self, re=None):
        u"""Retourne (finesse_max, alpha_finesse_max) pour un Re donne.

        :param re: Reynolds (None = premier disponible)
        :type re: float or None
        :returns: (finesse_max, alpha_finesse_max) ou None
        :rtype: tuple or None
        """
        polar = self.get_polar(re)
        if polar is None or 'CL' not in polar or 'CD' not in polar:
            return None
        cl = polar['CL']
        cd = polar['CD']
        alpha = polar['alpha']
        if len(cl) == 0:
            return None
        # Eviter la division par zero
        mask = cd > 1e-8
        if not np.any(mask):
            return None
        finesse = np.where(mask, cl / cd, 0.0)
        idx = np.argmax(finesse)
        return (float(finesse[idx]), float(alpha[idx]))

    # ------------------------------------------------------------------
    #  Methodes de trace — granulaires (composables)
    # ------------------------------------------------------------------

    # Couleur par defaut : bleu matplotlib
    DEFAULT_COLOR = '#1f77b4'

    # Styles de ligne par Reynolds (cyclique)
    _RE_LINESTYLES = ['-', '--', '-.', ':']

    @staticmethod
    def _format_re(re_val):
        u"""Formate un Reynolds pour la legende.

        :param re_val: nombre de Reynolds
        :type re_val: float
        :returns: chaine formatee (ex: 'Re=300k', 'Re=1M')
        :rtype: str
        """
        if re_val >= 1e6:
            return 'Re=%.0fM' % (re_val / 1e6)
        elif re_val >= 1e3:
            return 'Re=%.0fk' % (re_val / 1e3)
        else:
            return 'Re=%g' % re_val

    def _get_or_create_ax(self, ax, show):
        u"""Retourne l'ax fourni ou en cree un nouveau.

        :param ax: axes matplotlib ou None
        :param show: si True et ax cree, appeler plt.show() a la fin
        :returns: (ax, created) — created=True si nouvel ax
        """
        if ax is not None:
            return ax, False
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        return ax, True

    def plot_cl(self, ax=None, re=None, color=None, label=None, show=True):
        u"""Trace CL(alpha).

        :param ax: axes existants (None = nouveau)
        :param re: Reynolds (None = tous les Re disponibles)
        :param color: couleur (defaut : bleu)
        :param label: etiquette pour la legende
        :param show: appeler plt.show() si nouveau ax
        :returns: axes matplotlib
        """
        if color is None:
            color = self.DEFAULT_COLOR
        ax, created = self._get_or_create_ax(ax, show)
        re_vals = [re] if re is not None else self.re_list
        single = (len(re_vals) == 1)
        for i, re_val in enumerate(re_vals):
            polar = self.get_polar(re_val)
            if polar is None:
                continue
            ls = '-' if single else self._RE_LINESTYLES[i % len(self._RE_LINESTYLES)]
            re_tag = self._format_re(re_val)
            lbl = label if (label and single) else (
                '%s %s' % (label, re_tag) if label else re_tag)
            ax.plot(polar['alpha'], polar['CL'],
                    color=color, linestyle=ls, label=lbl)
        ax.set_xlabel(u'alpha (deg)')
        ax.set_ylabel(u'CL')
        ax.set_title(u'CL(alpha)')
        ax.grid(True)
        ax.legend(fontsize=8)
        if created and show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax

    def plot_cd(self, ax=None, re=None, color=None, label=None, show=True):
        u"""Trace CD(alpha).

        :param ax: axes existants (None = nouveau)
        :param re: Reynolds (None = tous les Re disponibles)
        :param color: couleur (defaut : bleu)
        :param label: etiquette pour la legende
        :param show: appeler plt.show() si nouveau ax
        :returns: axes matplotlib
        """
        if color is None:
            color = self.DEFAULT_COLOR
        ax, created = self._get_or_create_ax(ax, show)
        re_vals = [re] if re is not None else self.re_list
        single = (len(re_vals) == 1)
        for i, re_val in enumerate(re_vals):
            polar = self.get_polar(re_val)
            if polar is None:
                continue
            ls = '-' if single else self._RE_LINESTYLES[i % len(self._RE_LINESTYLES)]
            re_tag = self._format_re(re_val)
            lbl = label if (label and single) else (
                '%s %s' % (label, re_tag) if label else re_tag)
            ax.plot(polar['alpha'], polar['CD'],
                    color=color, linestyle=ls, label=lbl)
        ax.set_xlabel(u'alpha (deg)')
        ax.set_ylabel(u'CD')
        ax.set_title(u'CD(alpha)')
        ax.grid(True)
        ax.legend(fontsize=8)
        if created and show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax

    def plot_finesse(self, ax=None, re=None, color=None, label=None, show=True):
        u"""Trace CL/CD(alpha) — finesse.

        :param ax: axes existants (None = nouveau)
        :param re: Reynolds (None = tous les Re disponibles)
        :param color: couleur (defaut : bleu)
        :param label: etiquette pour la legende
        :param show: appeler plt.show() si nouveau ax
        :returns: axes matplotlib
        """
        if color is None:
            color = self.DEFAULT_COLOR
        ax, created = self._get_or_create_ax(ax, show)
        re_vals = [re] if re is not None else self.re_list
        single = (len(re_vals) == 1)
        for i, re_val in enumerate(re_vals):
            polar = self.get_polar(re_val)
            if polar is None:
                continue
            cl = polar['CL']
            cd = polar['CD']
            finesse = np.where(cd > 1e-8, cl / cd, 0.0)
            ls = '-' if single else self._RE_LINESTYLES[i % len(self._RE_LINESTYLES)]
            re_tag = self._format_re(re_val)
            lbl = label if (label and single) else (
                '%s %s' % (label, re_tag) if label else re_tag)
            ax.plot(polar['alpha'], finesse,
                    color=color, linestyle=ls, label=lbl)
        ax.set_xlabel(u'alpha (deg)')
        ax.set_ylabel(u'CL/CD')
        ax.set_title(u'Finesse(alpha)')
        ax.grid(True)
        ax.legend(fontsize=8)
        if created and show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax

    def plot_drag_polar(self, ax=None, re=None, color=None, label=None, show=True):
        u"""Trace CL(CD) — trainee polaire.

        :param ax: axes existants (None = nouveau)
        :param re: Reynolds (None = tous les Re disponibles)
        :param color: couleur (defaut : bleu)
        :param label: etiquette pour la legende
        :param show: appeler plt.show() si nouveau ax
        :returns: axes matplotlib
        """
        if color is None:
            color = self.DEFAULT_COLOR
        ax, created = self._get_or_create_ax(ax, show)
        re_vals = [re] if re is not None else self.re_list
        single = (len(re_vals) == 1)
        for i, re_val in enumerate(re_vals):
            polar = self.get_polar(re_val)
            if polar is None:
                continue
            ls = '-' if single else self._RE_LINESTYLES[i % len(self._RE_LINESTYLES)]
            re_tag = self._format_re(re_val)
            lbl = label if (label and single) else (
                '%s %s' % (label, re_tag) if label else re_tag)
            ax.plot(polar['CD'], polar['CL'],
                    color=color, linestyle=ls, label=lbl)
        ax.set_xlabel(u'CD')
        ax.set_ylabel(u'CL')
        ax.set_title(u'Trainee polaire')
        ax.grid(True)
        ax.legend(fontsize=8)
        if created and show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax

    def plot_cp(self, alpha, ax=None, re=None, color=None, label=None, show=True):
        u"""Trace Cp(x) pour un alpha donne.

        :param alpha: angle d'attaque
        :type alpha: float
        :param ax: axes existants (None = nouveau)
        :param re: Reynolds (None = tous les Re disponibles)
        :param color: couleur (defaut : bleu)
        :param label: etiquette pour la legende
        :param show: appeler plt.show() si nouveau ax
        :returns: axes matplotlib
        """
        if color is None:
            color = self.DEFAULT_COLOR
        ax, created = self._get_or_create_ax(ax, show)
        re_vals = [re] if re is not None else sorted(self._cp.keys())
        single = (len(re_vals) == 1)
        for i, re_val in enumerate(re_vals):
            if re_val not in self._cp:
                continue
            cp_data = self._cp[re_val].get(alpha, None)
            if cp_data is None:
                continue
            ls = '-' if single else self._RE_LINESTYLES[i % len(self._RE_LINESTYLES)]
            re_tag = self._format_re(re_val)
            lbl = label if (label and single) else (
                '%s %s' % (label, re_tag) if label else re_tag)
            ax.plot(cp_data[:, 0], cp_data[:, 1],
                    color=color, linestyle=ls, label=lbl)
        ax.set_xlabel(u'x/c')
        ax.set_ylabel(u'Cp')
        ax.set_title(u'Cp(x)  alpha=%.1f deg' % alpha)
        ax.invert_yaxis()
        ax.grid(True)
        ax.legend(fontsize=8)
        if created and show:
            import matplotlib.pyplot as plt
            plt.show()
        return ax

    # ------------------------------------------------------------------
    #  Methodes de trace — synthetiques
    # ------------------------------------------------------------------

    def plot_polars(self, fig=None, color=None, show=True):
        u"""Trace les 4 graphiques polaires (multi-Re).

        :param fig: figure matplotlib existante (None = nouvelle)
        :param color: couleur (defaut : bleu)
        :param show: appeler plt.show() si nouvelle figure
        :returns: figure matplotlib
        """
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.figure(figsize=(10, 7))

        ax_cl = fig.add_subplot(2, 2, 1)
        ax_cd = fig.add_subplot(2, 2, 2)
        ax_fin = fig.add_subplot(2, 2, 3)
        ax_clcd = fig.add_subplot(2, 2, 4)

        for re_val in self.re_list:
            label = 'Re=%g' % re_val
            self.plot_cl(ax=ax_cl, re=re_val, color=color, label=label, show=False)
            self.plot_cd(ax=ax_cd, re=re_val, color=color, label=label, show=False)
            self.plot_finesse(ax=ax_fin, re=re_val, color=color, label=label, show=False)
            self.plot_drag_polar(ax=ax_clcd, re=re_val, color=color, label=label, show=False)

        fig.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    #  Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return "SimulationResults(%d Re, %d pts converges, %d warnings)" % (
            len(self.re_list), self.n_converged, len(self._warnings))


# ======================================================================
#  Simulation
# ======================================================================

class Simulation(object):
    u"""Simulation aerodynamique 2D d'un profil.

    Lie un objet Profil a des parametres de simulation et orchestre
    le pipeline (Preprocessor -> Simulator -> Postprocessor).

    Usage::

        profil = Profil.from_naca('2412')
        sim = Simulation(profil, params={'RE': 500000})
        results = sim.run()
        print(results)
    """

    # Etats possibles
    IDLE = 'idle'
    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'

    def __init__(self, profil, params=None, solver='xfoil', work_dir=None):
        u"""
        :param profil: profil a analyser
        :type profil: Profil
        :param params: parametres utilisateur (surchargent les defauts)
        :type params: dict or None
        :param solver: nom du solveur
        :type solver: str
        :param work_dir: repertoire de travail (None = temporaire)
        :type work_dir: str or None
        """
        from profil import Profil
        if not isinstance(profil, Profil):
            raise TypeError(
                u"profil doit etre une instance de Profil, pas %s"
                % type(profil).__name__)

        self._profil = profil
        self._solver = solver
        self._state = self.IDLE
        self._results = None
        self._error = None

        # Repertoire de travail
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix='foil2d_sim_')
        self._work_dir = work_dir

        # Fusionner les parametres
        from foilconfig import load_defaults, merge_params
        try:
            defaults = load_defaults(solver)
        except IOError:
            defaults = {}
        self._params = merge_params(defaults, params)

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def profil(self):
        u"""Profil analyse (lecture seule)."""
        return self._profil

    @property
    def params(self):
        u"""Parametres de simulation (copie, lecture seule)."""
        return dict(self._params)

    @property
    def solver(self):
        u"""Nom du solveur."""
        return self._solver

    @property
    def state(self):
        u"""Etat de la simulation : 'idle', 'running', 'done', 'failed'."""
        return self._state

    @property
    def results(self):
        u"""Resultats de la simulation (SimulationResults ou None)."""
        return self._results

    @property
    def work_dir(self):
        u"""Repertoire de travail."""
        return self._work_dir

    @property
    def error(self):
        u"""Message d'erreur si state='failed', None sinon."""
        return self._error

    @property
    def is_done(self):
        u"""True si la simulation est terminee avec succes."""
        return self._state == self.DONE

    @property
    def is_failed(self):
        u"""True si la simulation a echoue."""
        return self._state == self.FAILED

    @property
    def has_results(self):
        u"""True si des resultats sont disponibles."""
        return self._results is not None

    # ------------------------------------------------------------------
    #  Methodes
    # ------------------------------------------------------------------

    def run(self):
        u"""Execute le pipeline complet.

        Normalise le profil en interne (copie de travail),
        passe les points au pipeline, encapsule les resultats.

        :returns: resultats de la simulation
        :rtype: SimulationResults
        :raises RuntimeError: si la simulation est deja en cours
        """
        if self._state == self.RUNNING:
            raise RuntimeError(u"Simulation deja en cours")

        self._state = self.RUNNING
        self._error = None

        try:
            from profil import Profil
            from pipeline import FoilAnalysisPipeline

            # Copier et normaliser le profil pour le solveur
            p_temp = Profil(self._profil.points.copy(),
                            name=self._profil.name)
            p_temp.normalize()
            # Convertir mm -> coordonnees normalisees [0, 1]
            pts_norm = p_temp.points / 1000.0

            logger.info(u"Simulation '%s' avec %s (Re=%s)",
                        self._profil.name, self._solver,
                        self._params.get('RE', '?'))

            # Creer et executer le pipeline
            pipeline = FoilAnalysisPipeline(
                solver=self._solver,
                work_dir=self._work_dir)
            raw_results = pipeline.run(pts_norm, self._params)

            # Encapsuler les resultats
            self._results = SimulationResults.from_dict(raw_results)
            self._state = self.DONE

            logger.info(u"Simulation terminee : %s", repr(self._results))
            return self._results

        except Exception as e:
            self._state = self.FAILED
            self._error = str(e)
            logger.error(u"Simulation echouee : %s", str(e))
            raise

    def prepare_only(self):
        u"""Execute uniquement le preprocessing (debug/inspection).

        :returns: liste des fichiers generes
        :rtype: list[str]
        """
        from profil import Profil
        from pipeline import FoilAnalysisPipeline

        p_temp = Profil(self._profil.points.copy(),
                        name=self._profil.name)
        p_temp.normalize()
        pts_norm = p_temp.points / 1000.0

        pipeline = FoilAnalysisPipeline(
            solver=self._solver,
            work_dir=self._work_dir)
        return pipeline.prepare_only(pts_norm, self._params)

    def reset(self):
        u"""Remet la simulation a l'etat initial.

        Efface les resultats et l'erreur.

        :returns: self
        :rtype: Simulation
        """
        self._state = self.IDLE
        self._results = None
        self._error = None
        return self

    # ------------------------------------------------------------------
    #  Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return "Simulation('%s', solver='%s', state='%s')" % (
            self._profil.name, self._solver, self._state)
