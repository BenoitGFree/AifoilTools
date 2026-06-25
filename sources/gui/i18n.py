#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internationalisation legere de la GUI AirfoilTools.

La chaine francaise sert de cle : ``tr(s)`` renvoie ``s`` tel quel en
francais, ou sa traduction anglaise. Les chaines non encore traduites
restent en francais (fallback gracieux), ce qui permet une couverture
progressive sans casser l'affichage.

Langue persistee via QSettings ; le changement prend effet au
redemarrage (pas de re-traduction live des widgets).
"""

from PySide6.QtCore import QSettings

#: Langues supportees (code -> libelle natif, affiche tel quel)
LANGUAGES = (('fr', u'Français'), ('en', u'English'))
_CODES = tuple(code for code, _name in LANGUAGES)
_DEFAULT = 'fr'

_lang = _DEFAULT


def load_language():
    """Charge la langue depuis QSettings et l'active. Retourne le code."""
    val = QSettings().value('language', _DEFAULT)
    set_language(val)
    return _lang


def save_language(lang):
    """Persiste la langue dans QSettings (sans changer la langue active)."""
    if lang in _CODES:
        QSettings().setValue('language', lang)


def set_language(lang):
    """Active la langue courante (en memoire)."""
    global _lang
    _lang = lang if lang in _CODES else _DEFAULT


def get_language():
    """Retourne le code de langue courant ('fr' ou 'en')."""
    return _lang


def tr(s):
    """Traduit une chaine. La cle est la source francaise.

    :param s: chaine source (francaise)
    :returns: la traduction dans la langue courante, ou ``s`` en
        fallback (francais ou cle absente du dictionnaire)
    """
    if _lang == 'fr':
        return s
    return _TR.get(_lang, {}).get(s, s)


# ----------------------------------------------------------------------
# Dictionnaire de traductions. Cle = source FR exacte (avec accents et
# mnemoniques '&'). Couverture progressive : tout ce qui manque ici
# reste affiche en francais.
# ----------------------------------------------------------------------
_TR = {
    'en': {
        # === Onglets / etat ===
        u"Profils": u"Airfoils",
        u"Paramétrage XFoil": u"XFoil Settings",
        u"Résultats": u"Results",
        u"Pret": u"Ready",

        # === Menu Fichier ===
        u"&Fichier": u"&File",
        u"Ouvrir profil &courant...": u"Open &current airfoil...",
        u"Charger un profil (.dat / .bspl / .bez / .csv) comme profil"
        u" courant (bleu)":
            u"Load an airfoil (.dat / .bspl / .bez / .csv) as the current"
            u" airfoil (blue)",
        u"Ouvrir profil référence...":
            u"Open &reference airfoil...",
        u"Charger un profil comme profil de reference (rouge)":
            u"Load an airfoil as the reference airfoil (red)",
        u"Depuis la base &UIUC": u"From the &UIUC database",
        u"Charger un profil depuis la base de donnees UIUC (Selig)":
            u"Load an airfoil from the UIUC database (Selig)",
        u"Profil courant…": u"Current airfoil…",
        u"Telecharger un profil depuis UIUC comme profil courant":
            u"Download an airfoil from UIUC as the current airfoil",
        u"Profil référence…": u"Reference airfoil…",
        u"Telecharger un profil depuis UIUC comme profil de reference":
            u"Download an airfoil from UIUC as the reference airfoil",
        u"Profil &NACA": u"&NACA airfoil",
        u"Generer un profil NACA (4 ou 5 chiffres) a partir de ses"
        u" indices":
            u"Generate a NACA airfoil (4 or 5 digits) from its indices",
        u"Generer un profil NACA comme profil courant":
            u"Generate a NACA airfoil as the current airfoil",
        u"Generer un profil NACA comme profil de reference":
            u"Generate a NACA airfoil as the reference airfoil",
        u"&Sauvegarder profil...": u"&Save airfoil...",
        u"Sauvegarder le profil courant (formats : Selig .dat,"
        u" Lednicer .dat, Spline .bspl, CSV)":
            u"Save the current airfoil (formats: Selig .dat, Lednicer"
            u" .dat, Spline .bspl, CSV)",
        u"Sauvegarder profil avec &volet...":
            u"Save airfoil with &flap...",
        u"Sauvegarder le profil avec volet braque (vert), de la meme"
        u" maniere que le profil courant (complet, extrados ou"
        u" intrados). Necessite le volet active.":
            u"Save the airfoil with deflected flap (green), the same way"
            u" as the current airfoil (full, upper or lower surface)."
            u" Requires the flap to be enabled.",
        u"Ouvrir &projet...": u"Open &project...",
        u"Ouvrir un projet AirfoilTools (.aftproj) : profils +"
        u" image de calque":
            u"Open an AirfoilTools project (.aftproj): airfoils +"
            u" tracing image",
        u"&Enregistrer projet...": u"Save pro&ject...",
        u"Enregistrer le projet (profils courant/reference +"
        u" image de calque) au format .aftproj":
            u"Save the project (current/reference airfoils + tracing"
            u" image) in .aftproj format",
        u"Charger &image de calque...": u"Load tracing &image...",
        u"Charger une image (jpg, png, tif...) en arriere-plan pour"
        u" decalquer un profil. Manipulation : touche « i »"
        u" + souris":
            u"Load an image (jpg, png, tif...) in the background to"
            u" trace an airfoil. Handling: « i » key + mouse",
        u"Retirer l'image de calque": u"Remove tracing image",
        u"Supprimer l'image de calque actuellement affichee":
            u"Remove the currently displayed tracing image",
        u"&Quitter": u"&Quit",
        u"Fermer l'application": u"Close the application",

        # === Menu Edition ===
        u"&Edition": u"&Edit",
        u"&Annuler": u"&Undo",
        u"Annuler la derniere action (non implemente)":
            u"Undo the last action (not implemented)",
        u"&Refaire": u"&Redo",
        u"Refaire l'action annulee (non implemente)":
            u"Redo the undone action (not implemented)",
        u"Convertir en Spline": u"Convert to Spline",
        u"Convertit le profil courant (mode points discrets) en"
        u" splines de Bezier multi-segment editables":
            u"Convert the current airfoil (discrete points mode) into"
            u" editable multi-segment Bezier splines",
        u"Échantillonnage": u"Sampling",
        u"Modifier le nombre de points d'echantillonnage des splines":
            u"Change the number of spline sampling points",
        u"Profil &courant...": u"&Current airfoil...",
        u"Choisir le nombre de points pour le profil courant"
        u" (necessite mode Spline)":
            u"Choose the number of points for the current airfoil"
            u" (requires Spline mode)",
        u"Profil référence...": u"Reference airfoil...",
        u"Choisir le nombre de points pour le profil de reference"
        u" (necessite mode Spline)":
            u"Choose the number of points for the reference airfoil"
            u" (requires Spline mode)",

        # === Menu Affichage ===
        u"&Affichage": u"&View",
        u"Zoom &adapte": u"&Fit zoom",
        u"Recadrer la vue sur l'ensemble des profils visibles":
            u"Fit the view to all visible airfoils",
        u"&Disposition": u"&Layout",
        u"Choisir la grille (lignes x colonnes) de l'onglet Resultats":
            u"Choose the grid (rows x columns) of the Results tab",

        # === Menu Options ===
        u"&Options": u"&Options",
        u"Déviation": u"Deviation",
        u"Mode de calcul de la deviation entre profil courant et"
        u" reference":
            u"Deviation computation mode between current and reference"
            u" airfoils",
        u"Verticale (épaisseur)": u"Vertical (thickness)",
        u"Ecart mesure verticalement (selon z), a abscisse constante":
            u"Gap measured vertically (along z), at constant abscissa",
        u"Normale (perpendiculaire)": u"Normal (perpendicular)",
        u"Ecart mesure perpendiculairement a la surface du profil"
        u" courant":
            u"Gap measured perpendicular to the current airfoil surface",
        u"&Langue": u"&Language",
        u"Choisir la langue de l'interface (effet au redemarrage)":
            u"Choose the interface language (takes effect on restart)",

        # === Menu Aide ===
        u"&Aide": u"&Help",
        u"&Manuel utilisateur": u"&User manual",
        u"Ouvrir le manuel utilisateur (PDF) avec le visualiseur"
        u" par defaut du systeme":
            u"Open the user manual (PDF) with the system default viewer",
        u"A &propos...": u"A&bout...",
        u"Informations sur AirfoilTools":
            u"Information about AirfoilTools",

        # === Boite de dialogue Langue / redemarrage ===
        u"Langue de l'interface": u"Interface language",
        u"La langue sera appliquee au prochain demarrage"
        u" d'AirfoilTools.":
            u"The language will be applied the next time AirfoilTools"
            u" starts.",

        # === A propos ===
        u"AirfoilTools - Analyse aérodynamique 2D\n"
        u"Courbes de Bézier, profils, XFoil\n\n":
            u"AirfoilTools - 2D aerodynamic analysis\n"
            u"Bezier curves, airfoils, XFoil\n\n",
        u"Première version : 2022\n\n":
            u"First release: 2022\n\n",
        u"Auteur : Benoît Gagnaire":
            u"Author: Benoît Gagnaire",

        # === Manuel : messages ===
        u"Manuel introuvable": u"Manual not found",
        u"Le fichier manuel.pdf n'a pas ete trouve.\n\n"
        u"Verifiez que le fichier docs/manuel/manuel.pdf "
        u"existe a cote de l'application.":
            u"The manual PDF file was not found.\n\n"
            u"Check that docs/manuel/manuel.pdf exists next to the"
            u" application.",
        u"Impossible d'ouvrir le manuel":
            u"Cannot open the manual",
        u"Manuel ouvert : %s": u"Manual opened: %s",
        u"Erreur lors de l'ouverture du PDF :\n\n%s":
            u"Error while opening the PDF:\n\n%s",

        # Noms d'analyses (resultats) : seul 'Finesse' n'est pas une
        # notation aero internationale. Identifiant interne inchange.
        u"Finesse(alpha)": u"L/D(alpha)",
        u"Finesse(CL)": u"L/D(CL)",
        u"Cp + Profil": u"Cp + Airfoil",

        # Onglet « Cp / Couche limite »
        u"Cp / Couche limite": u"Cp / Boundary layer",
        u"Cp + couche limite": u"Cp + boundary layer",
        u"Profil :": u"Airfoil:",
        u"Re :": u"Re:",
        u"α :": u"α:",
        u"Référence": u"Reference",
        u"Extrados": u"Upper",
        u"Intrados": u"Lower",
        u"Non visqueux": u"Inviscid",
        u"Profil": u"Airfoil",
        u"δ* extrados": u"δ* upper",
        u"δ* intrados": u"δ* lower",
        u"Calcul indisponible.": u"Calculation unavailable.",
        u"Pas de Cp ni de couche limite pour ce calcul.":
            u"No Cp nor boundary layer for this calculation.",
        u"Profil a visualiser : courant (bleu), reference (rouge)"
        u" ou volet (vert).":
            u"Airfoil to display: current (blue), reference (red)"
            u" or flap (green).",
        u"Nombre de Reynolds du calcul.":
            u"Reynolds number of the calculation.",
        u"Incidence (degres) du calcul.":
            u"Angle of attack (degrees) of the calculation.",
    },
}

# Traductions Phase 2 (onglets / dialogues), generees dans i18n_en.py
# par tools/i18n_assemble.py. Fusionnees ici dans la table EN.
from .i18n_en import EN_PHASE2 as _EN_PHASE2  # noqa: E402

_TR['en'].update(_EN_PHASE2)
