#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Objet Analyse : post-traitement et traces comparatifs de simulations.

Prend en entree des objets ``Simulation`` (deja executes ou non)
et fournit des methodes de comparaison : tableaux recapitulatifs,
traces polaires, distributions Cp.

Usage::

    from model.aerodynamique.foil2d.profil import Profil
    from model.aerodynamique.foil2d.simulation import Simulation
    from model.aerodynamique.foil2d.analyse import Analyse

    sim_ref = Simulation(Profil.from_naca('0012'), params={...})
    sim_new = Simulation(Profil.from_naca('2412'), params={...})
    sim_ref.run()
    sim_new.run()

    analyse = Analyse()
    analyse.add(sim_ref, reference=True)
    analyse.add(sim_new)
    analyse.summary()
    analyse.plot_polars()
    analyse.plot_cp(5.0)

@author: Nervures
@date: 2026-02
"""


class Analyse(object):
    u"""Post-traitement et traces comparatifs de simulations.

    Rassemble des objets ``Simulation`` et fournit :
    - un tableau recapitulatif (``summary()``)
    - des traces comparatifs (``plot_polars()``, ``plot_cp()``, etc.)

    Un objet Simulation peut etre designe comme **reference** (trace en rouge).
    Les autres utilisent une palette de couleurs distinctes.
    Les differents Reynolds sont differencies par le style de ligne.
    """

    # Couleur de la reference
    REF_COLOR = 'red'

    # Palette pour les simulations non-reference (cyclique)
    _COLORS = [
        '#1f77b4',  # bleu
        '#2ca02c',  # vert
        '#ff7f0e',  # orange
        '#9467bd',  # violet
        '#8c564b',  # marron
        '#e377c2',  # rose
        '#7f7f7f',  # gris
        '#bcbd22',  # jaune-vert
    ]

    def __init__(self):
        u"""Cree une analyse vide."""
        self._entries = []

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def add(self, simulation, name=None, reference=False):
        u"""Ajoute une simulation a l'analyse.

        :param simulation: simulation (deja executee ou non)
        :type simulation: Simulation
        :param name: nom pour la legende (defaut: simulation.profil.name)
        :type name: str or None
        :param reference: si True, cette simulation est la reference (rouge)
        :type reference: bool
        :returns: self (chainage)
        :rtype: Analyse
        """
        if name is None:
            name = simulation.profil.name

        # Si reference, desactiver l'ancienne
        if reference:
            for e in self._entries:
                e['reference'] = False

        self._entries.append({
            'name': name,
            'reference': reference,
            'simulation': simulation,
        })

        self._reassign_colors()
        return self

    def set_reference(self, name):
        u"""Designe une simulation existante comme reference.

        :param name: nom de la simulation
        :type name: str
        :returns: self
        :rtype: Analyse
        :raises KeyError: si le nom n'existe pas
        """
        found = False
        for e in self._entries:
            e['reference'] = (e['name'] == name)
            if e['reference']:
                found = True
        if not found:
            raise KeyError(u"Simulation '%s' non trouvee" % name)
        self._reassign_colors()
        return self

    def _reassign_colors(self):
        u"""Reassigne les couleurs apres modification."""
        i_color = 0
        for e in self._entries:
            if e['reference']:
                e['color'] = self.REF_COLOR
            else:
                e['color'] = self._COLORS[i_color % len(self._COLORS)]
                i_color += 1

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def names(self):
        u"""Noms des simulations dans l'ordre d'ajout."""
        return [e['name'] for e in self._entries]

    @property
    def reference(self):
        u"""L'entry marquee comme reference, ou None."""
        for e in self._entries:
            if e['reference']:
                return e
        return None

    # ------------------------------------------------------------------
    #  Acces
    # ------------------------------------------------------------------

    def get(self, name):
        u"""Retourne l'entry par nom.

        :param name: nom de la simulation
        :type name: str
        :returns: dict {'name', 'simulation', 'reference', 'color'}
        :rtype: dict
        :raises KeyError: si absent
        """
        for e in self._entries:
            if e['name'] == name:
                return e
        raise KeyError(u"Simulation '%s' non trouvee" % name)

    def results(self, name):
        u"""Retourne les resultats d'une simulation par nom.

        :param name: nom de la simulation
        :type name: str
        :returns: resultats ou None
        :rtype: SimulationResults or None
        """
        return self.get(name)['simulation'].results

    # ------------------------------------------------------------------
    #  Recapitulatif
    # ------------------------------------------------------------------

    def summary(self):
        u"""Affiche un tableau recapitulatif des resultats.

        Pour chaque simulation et chaque Reynolds : CL_max, finesse_max.
        """
        from .simulation import SimulationResults
        for e in self._entries:
            sim = e['simulation']
            name = e['name']
            ref_tag = ' (ref)' if e['reference'] else ''
            print("\n=== %s%s ===" % (name, ref_tag))
            if not sim.has_results:
                print("  Pas de resultats (state=%s)" % sim.state)
                continue
            r = sim.results
            print("  %d Re, %d pts converges"
                  % (len(r.re_list), r.n_converged))
            for re_val in r.re_list:
                cl_info = r.cl_max(re_val)
                f_info = r.finesse_max(re_val)
                re_tag = SimulationResults._format_re(re_val)
                parts = ["  %s :" % re_tag]
                if cl_info:
                    parts.append("CL_max=%.3f (a=%.1f)" % cl_info)
                if f_info:
                    parts.append("f_max=%.1f (a=%.1f)" % f_info)
                print("  ".join(parts))

    # ------------------------------------------------------------------
    #  Ordre de trace et titres
    # ------------------------------------------------------------------

    def _ordered_entries(self):
        u"""Retourne les entries : reference en premier, puis les autres."""
        ref = []
        others = []
        for e in self._entries:
            if e['reference']:
                ref.append(e)
            else:
                others.append(e)
        return ref + others

    def _make_title(self):
        u"""Genere un titre automatique."""
        names = [e['name'] for e in self._entries]
        ref = self.reference
        if len(names) == 1:
            return names[0]
        elif len(names) == 2 and ref:
            other = [n for n in names if n != ref['name']]
            return u'%s vs %s (ref)' % (other[0], ref['name'])
        else:
            return u'Comparaison de %d profils' % len(names)

    def _label(self, entry):
        u"""Label pour la legende, avec '(ref)' si reference."""
        if entry['reference']:
            return entry['name'] + ' (ref)'
        return entry['name']

    # ------------------------------------------------------------------
    #  Methodes de trace
    # ------------------------------------------------------------------

    def plot_polars(self, show=True):
        u"""Trace les 4 graphiques polaires comparatifs.

        CL(alpha), CD(alpha), Finesse(alpha), Trainee polaire CL(CD).
        Tous les profils superposes. Reference en rouge.

        :param show: appeler plt.show()
        :type show: bool
        :returns: figure matplotlib
        """
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(self._make_title(), fontsize=13)

        for entry in self._ordered_entries():
            r = entry['simulation'].results
            if r is None:
                continue
            lbl = self._label(entry)
            col = entry['color']
            r.plot_cl(ax=axes[0, 0], color=col, label=lbl, show=False)
            r.plot_cd(ax=axes[0, 1], color=col, label=lbl, show=False)
            r.plot_finesse(ax=axes[1, 0], color=col, label=lbl, show=False)
            r.plot_drag_polar(ax=axes[1, 1], color=col, label=lbl, show=False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if show:
            plt.show()
        return fig

    def plot_cp(self, alpha, show=True):
        u"""Trace Cp(x) comparatif a un alpha donne.

        :param alpha: angle d'attaque
        :type alpha: float
        :param show: appeler plt.show()
        :type show: bool
        :returns: figure matplotlib
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(u'%s — Cp a alpha=%.1f deg'
                     % (self._make_title(), alpha), fontsize=13)

        for entry in self._ordered_entries():
            r = entry['simulation'].results
            if r is None:
                continue
            r.plot_cp(alpha, ax=ax, color=entry['color'],
                      label=self._label(entry), show=False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if show:
            plt.show()
        return fig

    def plot_cl(self, show=True):
        u"""Trace CL(alpha) comparatif.

        :param show: appeler plt.show()
        :type show: bool
        :returns: figure matplotlib
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(u'%s — CL(alpha)' % self._make_title(), fontsize=13)

        for entry in self._ordered_entries():
            r = entry['simulation'].results
            if r is None:
                continue
            r.plot_cl(ax=ax, color=entry['color'],
                      label=self._label(entry), show=False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if show:
            plt.show()
        return fig

    def plot_cd(self, show=True):
        u"""Trace CD(alpha) comparatif.

        :param show: appeler plt.show()
        :type show: bool
        :returns: figure matplotlib
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(u'%s — CD(alpha)' % self._make_title(), fontsize=13)

        for entry in self._ordered_entries():
            r = entry['simulation'].results
            if r is None:
                continue
            r.plot_cd(ax=ax, color=entry['color'],
                      label=self._label(entry), show=False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if show:
            plt.show()
        return fig

    def plot_finesse(self, show=True):
        u"""Trace Finesse(alpha) comparatif.

        :param show: appeler plt.show()
        :type show: bool
        :returns: figure matplotlib
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(u'%s — Finesse(alpha)' % self._make_title(),
                     fontsize=13)

        for entry in self._ordered_entries():
            r = entry['simulation'].results
            if r is None:
                continue
            r.plot_finesse(ax=ax, color=entry['color'],
                           label=self._label(entry), show=False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if show:
            plt.show()
        return fig

    def plot_drag_polar(self, show=True):
        u"""Trace la trainee polaire CL(CD) comparatif.

        :param show: appeler plt.show()
        :type show: bool
        :returns: figure matplotlib
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(u'%s — Trainee polaire' % self._make_title(),
                     fontsize=13)

        for entry in self._ordered_entries():
            r = entry['simulation'].results
            if r is None:
                continue
            r.plot_drag_polar(ax=ax, color=entry['color'],
                              label=self._label(entry), show=False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    #  Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        n = len(self._entries)
        ref = self.reference
        ref_str = ", ref='%s'" % ref['name'] if ref else ''
        n_done = sum(1 for e in self._entries if e['simulation'].is_done)
        n_idle = n - n_done
        parts = []
        if n_done:
            parts.append('%d done' % n_done)
        if n_idle:
            parts.append('%d idle' % n_idle)
        state_str = ', '.join(parts)
        return "Analyse(%d profils%s, state: %s)" % (n, ref_str, state_str)
