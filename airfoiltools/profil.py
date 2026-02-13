#!/usr/bin/python
#-*-coding: utf-8 -*-

u"""
Objet metier Profil aerodynamique 2D.

Encapsule les coordonnees d'un profil avec ses proprietes geometriques
calculees (corde, calage, epaisseur, cambrure, BA, BF) et des methodes
de transformation (scale, rotate, translate, normalize).

Creation::

    # Depuis un fichier
    p = Profil.from_file('naca2412.dat', fmt='selig', unit='mm')

    # Depuis une designation NACA
    p = Profil.from_naca('2412', n_points=200)

    # Direct
    p = Profil(points, name='MonProfil')

@author: Nervures
@date: 2026-02
"""

import os
import math
import logging

import numpy as np

try:
    from bezier import Bezier
except ImportError:
    from airfoiltools.bezier import Bezier

logger = logging.getLogger(__name__)


class Profil(object):
    u"""Profil aerodynamique 2D.

    Stockage interne des coordonnees en millimetres.
    Convention Selig : BF -> extrados -> BA -> intrados -> BF.
    """

    def __init__(self, points, name='Sans nom'):
        u"""
        :param points: coordonnees (x, y) en millimetres, shape (n, 2)
        :type points: numpy.ndarray or list
        :param name: nom du profil
        :type name: str
        """
        self._points = np.asarray(points, dtype=float)
        if self._points.ndim != 2 or self._points.shape[1] != 2:
            raise ValueError(
                u"points doit etre un tableau (n, 2), recu shape %s"
                % str(self._points.shape))
        self._name = name
        self._output_path = None
        self._output_format = 'selig'
        self._bezier_extrados = None   # Bezier or None
        self._bezier_intrados = None   # Bezier or None

    # ------------------------------------------------------------------
    #  Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return "Profil('%s', %d pts)" % (self._name, len(self._points))

    # ------------------------------------------------------------------
    #  Properties : attributs stockes
    # ------------------------------------------------------------------

    @property
    def name(self):
        u"""Nom du profil."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def points(self):
        u"""Coordonnees (x, y) en millimetres, ndarray(n, 2).

        En mode Bezier, reconstruit le profil complet (convention Selig)
        a partir des courbes extrados et intrados.
        """
        if self.has_beziers:
            ext = self.extrados    # BA -> BF
            intr = self.intrados   # BA -> BF
            return np.vstack([ext[::-1], intr[1:]])  # BF->ext->BA->int->BF
        return self._points

    @points.setter
    def points(self, value):
        pts = np.asarray(value, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                u"points doit etre un tableau (n, 2), recu shape %s"
                % str(pts.shape))
        self._points = pts
        self._bezier_extrados = None
        self._bezier_intrados = None

    @property
    def output_path(self):
        u"""Chemin pour ecriture (str ou None)."""
        return self._output_path

    @output_path.setter
    def output_path(self, value):
        self._output_path = value

    @property
    def output_format(self):
        u"""Format d'ecriture ('selig', 'lednicer', 'csv')."""
        return self._output_format

    @output_format.setter
    def output_format(self, value):
        if value not in ('selig', 'lednicer', 'csv'):
            raise ValueError(
                u"Format inconnu '%s'. Attendu : selig, lednicer, csv" % value)
        self._output_format = value

    # ------------------------------------------------------------------
    #  Properties : geometrie calculee
    # ------------------------------------------------------------------

    @property
    def leading_edge(self):
        u"""Point de bord d'attaque (x, y) en mm — point de x minimal."""
        return self.find_leading_edge()

    @property
    def trailing_edge(self):
        u"""Point de bord de fuite (x, y) en mm."""
        return self.find_trailing_edge()

    @property
    def chord(self):
        u"""Longueur de la corde en mm (distance BA - BF)."""
        le = self.leading_edge
        te = self.trailing_edge
        return float(np.linalg.norm(te - le))

    @property
    def calage(self):
        u"""Calage du profil en degres (positif = nez en haut)."""
        le = self.leading_edge
        te = self.trailing_edge
        dx = te[0] - le[0]
        dy = te[1] - le[1]
        return -math.degrees(math.atan2(dy, dx))

    @property
    def relative_thickness(self):
        u"""Epaisseur relative maximale (e/c), sans unite."""
        return self.compute_thickness()

    @property
    def relative_camber(self):
        u"""Cambrure relative maximale (f/c), sans unite."""
        return self.compute_camber()

    @property
    def is_normalized(self):
        u"""True si BA~(0,0), corde~1000mm, calage~0 deg."""
        le = self.leading_edge
        te = self.trailing_edge
        c = self.chord
        inc = self.calage
        tol_pos = 1.0      # 1 mm
        tol_chord = 1.0     # 1 mm
        tol_angle = 0.1     # 0.1 deg
        return (abs(le[0]) < tol_pos
                and abs(le[1]) < tol_pos
                and abs(c - 1000.0) < tol_chord
                and abs(inc) < tol_angle)

    @property
    def has_beziers(self):
        u"""True si le profil est defini par des courbes de Bezier."""
        return (self._bezier_extrados is not None
                and self._bezier_intrados is not None)

    @property
    def bezier_extrados(self):
        u"""Courbe de Bezier de l'extrados (Bezier ou None)."""
        return self._bezier_extrados

    @property
    def bezier_intrados(self):
        u"""Courbe de Bezier de l'intrados (Bezier ou None)."""
        return self._bezier_intrados

    @property
    def extrados(self):
        u"""Points de l'extrados, du BA au BF (x croissant), ndarray(n, 2)."""
        if self._bezier_extrados is not None:
            return self._bezier_extrados.points
        i_ba = self._leading_edge_index()
        return self._points[:i_ba + 1][::-1].copy()

    @property
    def intrados(self):
        u"""Points de l'intrados, du BA au BF (x croissant), ndarray(n, 2)."""
        if self._bezier_intrados is not None:
            return self._bezier_intrados.points
        i_ba = self._leading_edge_index()
        return self._points[i_ba:].copy()

    # ------------------------------------------------------------------
    #  Methodes de calcul geometrique
    # ------------------------------------------------------------------

    def find_trailing_edge(self):
        u"""Trouve le bord de fuite : milieu du premier et dernier point.

        :returns: coordonnees (x, y) du BF
        :rtype: numpy.ndarray, shape (2,)
        """
        return 0.5 * (self._points[0] + self._points[-1])

    def _leading_edge_index(self):
        u"""Indice du bord d'attaque dans le tableau de points.

        :returns: indice du BA (point le plus eloigne du BF)
        :rtype: int
        """
        te = self.find_trailing_edge()
        dists = np.sum((self._points - te)**2, axis=1)
        return int(np.argmax(dists))

    def find_leading_edge(self):
        u"""Trouve le bord d'attaque : point le plus eloigne du bord de fuite.

        :returns: coordonnees (x, y) du BA
        :rtype: numpy.ndarray, shape (2,)
        """
        return self._points[self._leading_edge_index()].copy()

    def compute_thickness(self):
        u"""Calcule l'epaisseur relative maximale.

        Interpole extrados et intrados sur une grille commune en x,
        puis cherche l'ecart max perpendiculaire a la corde.

        :returns: epaisseur relative (e/c)
        :rtype: float
        """
        pts = self._points
        le = self.find_leading_edge()
        c = self.chord
        if c < 1e-12:
            return 0.0

        # Separer extrados / intrados au BA
        i_le = np.argmin(pts[:, 0])
        extrados = pts[:i_le + 1][::-1]   # BA -> BF, x croissant
        intrados = pts[i_le:]              # BA -> BF, x croissant

        if len(extrados) < 2 or len(intrados) < 2:
            return 0.0

        # Grille commune normalisee
        x_min = max(extrados[0, 0], intrados[0, 0])
        x_max = min(extrados[-1, 0], intrados[-1, 0])
        n_grid = 200
        x_grid = np.linspace(x_min, x_max, n_grid)

        y_ext = np.interp(x_grid, extrados[:, 0], extrados[:, 1])
        y_int = np.interp(x_grid, intrados[:, 0], intrados[:, 1])

        thickness = np.max(np.abs(y_ext - y_int))
        return float(thickness / c)

    def compute_camber(self):
        u"""Calcule la cambrure relative maximale.

        La ligne de cambrure est la moyenne de l'extrados et de l'intrados.
        La cambrure relative est l'ecart max de cette ligne a la corde.

        :returns: cambrure relative (f/c)
        :rtype: float
        """
        pts = self._points
        le = self.find_leading_edge()
        te = self.find_trailing_edge()
        c = self.chord
        if c < 1e-12:
            return 0.0

        i_le = np.argmin(pts[:, 0])
        extrados = pts[:i_le + 1][::-1]
        intrados = pts[i_le:]

        if len(extrados) < 2 or len(intrados) < 2:
            return 0.0

        x_min = max(extrados[0, 0], intrados[0, 0])
        x_max = min(extrados[-1, 0], intrados[-1, 0])
        n_grid = 200
        x_grid = np.linspace(x_min, x_max, n_grid)

        y_ext = np.interp(x_grid, extrados[:, 0], extrados[:, 1])
        y_int = np.interp(x_grid, intrados[:, 0], intrados[:, 1])

        # Ligne de cambrure
        y_camber = 0.5 * (y_ext + y_int)

        # Distance de la ligne de cambrure a la corde (droite BA-BF)
        # Droite BA-BF : direction d = te - le
        d = te - le
        d_norm = d / np.linalg.norm(d)
        n = np.array([-d_norm[1], d_norm[0]])  # normale a la corde

        # Ecart de chaque point cambrure par rapport a la droite BA-BF
        max_camber = 0.0
        for i in range(n_grid):
            p = np.array([x_grid[i], y_camber[i]])
            dist = abs(np.dot(p - le, n))
            if dist > max_camber:
                max_camber = dist

        return float(max_camber / c)

    # ------------------------------------------------------------------
    #  Methodes de transformation (in-place, retournent self)
    # ------------------------------------------------------------------

    def scale(self, factor):
        u"""Homothetie centree sur le bord d'attaque.

        :param factor: facteur d'echelle
        :type factor: float
        :returns: self (pour chainage)
        :rtype: Profil
        """
        le = self.find_leading_edge()
        self._points = le + (self._points - le) * factor
        return self

    def rotate(self, angle_deg, center=None):
        u"""Rotation du profil.

        :param angle_deg: angle en degres (positif = horaire = nez en haut)
        :type angle_deg: float
        :param center: centre de rotation (defaut : bord d'attaque)
        :type center: numpy.ndarray or None
        :returns: self
        :rtype: Profil
        """
        if center is None:
            center = self.find_leading_edge()
        center = np.asarray(center, dtype=float)
        a = math.radians(-angle_deg)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        pts = self._points - center
        x_new = pts[:, 0] * cos_a - pts[:, 1] * sin_a
        y_new = pts[:, 0] * sin_a + pts[:, 1] * cos_a
        self._points = np.column_stack([x_new, y_new]) + center
        return self

    def translate(self, dx, dy):
        u"""Translation du profil.

        :param dx: deplacement en x (mm)
        :type dx: float
        :param dy: deplacement en y (mm)
        :type dy: float
        :returns: self
        :rtype: Profil
        """
        self._points[:, 0] += dx
        self._points[:, 1] += dy
        return self

    def normalize(self):
        u"""Normalise le profil : BA en (0,0), corde = 1000 mm, calage = 0 deg.

        :returns: self
        :rtype: Profil
        """
        origin = np.array([0.0, 0.0])
        for _ in range(5):
            # Translater le BA en (0, 0)
            le = self.find_leading_edge()
            if np.linalg.norm(le) > 0.001:
                self.translate(-le[0], -le[1])

            # Annuler le calage (rotation centree sur l'origine)
            inc = self.calage
            if abs(inc) > 0.001:
                self.rotate(-inc, center=origin)

            # Mise a l'echelle : corde -> 1000 mm
            c = self.chord
            if c > 1e-12 and abs(c - 1000.0) > 0.001:
                self._points = self._points * (1000.0 / c)

            if self.is_normalized:
                break

        return self

    # ------------------------------------------------------------------
    #  Mode Bezier
    # ------------------------------------------------------------------

    def approximate_bezier(self, degree, max_dev=None, n_points=None):
        u"""Approxime extrados et intrados par des courbes de Bezier.

        Utilise les points discrets (``_points``) comme cibles.
        Les Beziers resultantes sont orientees du BA vers le BF.

        :param degree: degre des Beziers, ou ``'find'`` pour recherche auto
        :type degree: int or str
        :param max_dev: deviation max toleree (requis si degree='find')
        :type max_dev: float or None
        :param n_points: nombre de points d'echantillonnage des Beziers
            (defaut : nombre de points discrets de chaque cote)
        :type n_points: int or None
        :returns: self (pour chainage)
        :rtype: Profil
        """
        i_ba = self._leading_edge_index()
        ext_pts = self._points[:i_ba + 1][::-1]  # BA -> BF
        int_pts = self._points[i_ba:]             # BA -> BF

        n_ext = n_points if n_points is not None else len(ext_pts)
        n_int = n_points if n_points is not None else len(int_pts)

        self._bezier_extrados = Bezier(
            ext_pts, degree=degree, max_dev=max_dev, n_points=n_ext)
        self._bezier_intrados = Bezier(
            int_pts, degree=degree, max_dev=max_dev, n_points=n_int)
        return self

    def clear_beziers(self):
        u"""Supprime les courbes de Bezier, retour au mode points discrets.

        :returns: self (pour chainage)
        :rtype: Profil
        """
        self._bezier_extrados = None
        self._bezier_intrados = None
        return self

    # ------------------------------------------------------------------
    #  Methodes de creation (classmethods)
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, filepath, fmt='auto', unit='mm'):
        u"""Charge un profil depuis un fichier.

        :param filepath: chemin du fichier
        :type filepath: str
        :param fmt: format ('selig', 'lednicer', 'csv', 'auto')
        :type fmt: str
        :param unit: unite des coordonnees ('mm', 'm', 'normalized')
        :type unit: str
        :returns: instance Profil
        :rtype: Profil
        """
        filepath = str(filepath)
        if not os.path.isfile(filepath):
            raise IOError(u"Fichier introuvable : %s" % filepath)

        if fmt == 'auto':
            fmt = cls._detect_format(filepath)

        readers = {
            'selig': cls._read_selig,
            'lednicer': cls._read_lednicer,
            'csv': cls._read_csv,
        }
        if fmt not in readers:
            raise ValueError(
                u"Format inconnu '%s'. Attendu : %s"
                % (fmt, ', '.join(sorted(readers.keys()))))

        name, points = readers[fmt](filepath)
        points = cls._convert_units(points, unit)

        profil = cls(points, name=name)
        profil._output_path = filepath
        profil._output_format = fmt
        return profil

    @classmethod
    def from_naca(cls, designation, n_points=200):
        u"""Genere un profil NACA.

        :param designation: designation NACA (ex: '2412', '0012', '23012')
        :type designation: str
        :param n_points: nombre de points total
        :type n_points: int
        :returns: instance Profil (normalise, corde = 1000 mm)
        :rtype: Profil
        """
        designation = str(designation).strip()
        if len(designation) == 4:
            points = cls._naca_4digits(designation, n_points)
        elif len(designation) == 5:
            points = cls._naca_5digits(designation, n_points)
        else:
            raise ValueError(
                u"Designation NACA non supportee : '%s'. "
                u"Attendu : 4 ou 5 chiffres." % designation)

        # Points generes en coordonnees normalisees [0, 1]
        # Conversion en mm : corde = 1000 mm
        points = points * 1000.0

        name = 'NACA %s' % designation
        return cls(points, name=name)

    # ------------------------------------------------------------------
    #  Methodes de lecture (privees, statiques)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_format(filepath):
        u"""Detecte le format d'un fichier profil.

        Heuristiques :
        - CSV : contient des ';' ou des ','
        - Lednicer : 2e et 3e lignes sont des entiers (nb points ext/int)
        - Selig : par defaut

        :param filepath: chemin du fichier
        :type filepath: str
        :returns: format detecte
        :rtype: str
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        if not lines:
            return 'selig'

        # CSV : separateur ; ou , dans les lignes de donnees
        for line in lines[1:6]:
            if ';' in line:
                return 'csv'

        # Lednicer : lignes 2-3 contiennent un nombre de points
        if len(lines) >= 3:
            try:
                parts2 = lines[1].strip().split()
                parts3 = lines[2].strip().split()
                if (len(parts2) == 1 and len(parts3) == 1
                        and float(parts2[0]) > 1
                        and float(parts3[0]) > 1):
                    return 'lednicer'
            except (ValueError, IndexError):
                pass

        return 'selig'

    @staticmethod
    def _read_selig(filepath):
        u"""Lecture format Selig.

        Premiere ligne = nom du profil.
        Lignes suivantes = x y (espaces).

        :param filepath: chemin du fichier
        :type filepath: str
        :returns: (nom, points ndarray(n,2))
        :rtype: tuple
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        name = lines[0].strip() if lines else 'Sans nom'
        data = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    data.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue

        if not data:
            raise ValueError(
                u"Aucun point lu dans %s (format Selig)" % filepath)
        return name, np.array(data)

    @staticmethod
    def _read_lednicer(filepath):
        u"""Lecture format Lednicer.

        Ligne 1 = nom.
        Ligne 2 = nb points extrados (ex: '33.  33.')
        Ligne 3 = vide
        Lignes suivantes = extrados (x y), puis ligne vide, puis intrados (x y).

        :param filepath: chemin du fichier
        :type filepath: str
        :returns: (nom, points ndarray(n,2))
        :rtype: tuple
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        name = lines[0].strip() if lines else 'Sans nom'

        # Ligne 2 : compteur "nb_ext.  nb_int." -> sauter
        # On commence a parser apres la ligne de comptage
        data_lines = lines[2:]

        # Collecter les blocs de donnees separes par des lignes vides
        blocks = []
        current = []
        for line in data_lines:
            line = line.strip()
            if not line:
                if current:
                    blocks.append(current)
                    current = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    current.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
        if current:
            blocks.append(current)

        if len(blocks) < 2:
            raise ValueError(
                u"Format Lednicer invalide dans %s : "
                u"attendu 2 blocs (extrados + intrados)" % filepath)

        extrados = np.array(blocks[0])   # BA -> BF (x croissant)
        intrados = np.array(blocks[1])   # BA -> BF (x croissant)

        # Convention Selig : BF -> extrados(x decroissant) -> BA -> intrados(x croissant) -> BF
        points = np.vstack([extrados[::-1], intrados[1:]])
        return name, points

    @staticmethod
    def _read_csv(filepath):
        u"""Lecture format CSV (separateur ; ou ,).

        Premiere ligne = en-tete (ignoree si non numerique).
        Lignes suivantes = x;y ou x,y.

        :param filepath: chemin du fichier
        :type filepath: str
        :returns: (nom, points ndarray(n,2))
        :rtype: tuple
        """
        name = os.path.splitext(os.path.basename(filepath))[0]

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Detecter le separateur
        sep = ';' if ';' in ''.join(lines[:5]) else ','

        data = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(sep)
            if len(parts) >= 2:
                try:
                    data.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue  # en-tete ou ligne invalide

        if not data:
            raise ValueError(
                u"Aucun point lu dans %s (format CSV)" % filepath)
        return name, np.array(data)

    @staticmethod
    def _convert_units(points, unit):
        u"""Convertit les coordonnees vers millimetres.

        :param points: coordonnees brutes
        :type points: numpy.ndarray
        :param unit: unite source ('mm', 'm', 'normalized')
        :type unit: str
        :returns: coordonnees en mm
        :rtype: numpy.ndarray
        """
        if unit == 'mm':
            return points
        elif unit == 'm':
            return points * 1000.0
        elif unit == 'normalized':
            return points * 1000.0
        else:
            raise ValueError(
                u"Unite inconnue '%s'. Attendu : mm, m, normalized" % unit)

    # ------------------------------------------------------------------
    #  Generation NACA (methodes statiques privees)
    # ------------------------------------------------------------------

    @staticmethod
    def _naca_4digits(designation, n_points):
        u"""Genere un profil NACA 4 chiffres.

        :param designation: 4 caracteres (ex: '2412')
        :type designation: str
        :param n_points: nombre total de points
        :type n_points: int
        :returns: coordonnees normalisees (corde = 1), ndarray(n, 2)
        :rtype: numpy.ndarray
        """
        m = int(designation[0]) / 100.0    # cambrure max
        p = int(designation[1]) / 10.0     # position de cambrure max
        t = int(designation[2:4]) / 100.0  # epaisseur relative

        # Demi-profil : distribution cosinus pour resserrer au BA
        n_half = n_points // 2
        beta = np.linspace(0, np.pi, n_half)
        x = 0.5 * (1.0 - np.cos(beta))

        # Epaisseur (formule NACA)
        yt = 5.0 * t * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )

        # Ligne de cambrure
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
        if m > 0 and p > 0:
            front = x <= p
            back = ~front
            yc[front] = (m / p**2) * (2.0 * p * x[front] - x[front]**2)
            yc[back] = (m / (1.0 - p)**2) * (
                1.0 - 2.0 * p + 2.0 * p * x[back] - x[back]**2)
            dyc[front] = (2.0 * m / p**2) * (p - x[front])
            dyc[back] = (2.0 * m / (1.0 - p)**2) * (p - x[back])

        theta = np.arctan(dyc)

        # Extrados et intrados
        x_ext = x - yt * np.sin(theta)
        y_ext = yc + yt * np.cos(theta)
        x_int = x + yt * np.sin(theta)
        y_int = yc - yt * np.cos(theta)

        # Convention Selig : BF -> extrados (x decroissant) -> BA
        #                    -> intrados (x croissant) -> BF
        extrados = np.column_stack([x_ext[::-1], y_ext[::-1]])  # BF -> BA
        intrados = np.column_stack([x_int[1:], y_int[1:]])      # BA+1 -> BF

        return np.vstack([extrados, intrados])

    @staticmethod
    def _naca_5digits(designation, n_points):
        u"""Genere un profil NACA 5 chiffres.

        Supporte la serie standard (ex: '23012').

        :param designation: 5 caracteres (ex: '23012')
        :type designation: str
        :param n_points: nombre total de points
        :type n_points: int
        :returns: coordonnees normalisees (corde = 1), ndarray(n, 2)
        :rtype: numpy.ndarray
        """
        # Coefficients de la ligne de cambrure NACA 5 chiffres
        # l = premier chiffre, p_code = deuxieme chiffre, q = troisieme
        l_val = int(designation[0])
        p_code = int(designation[1])
        q = int(designation[2])
        t = int(designation[3:5]) / 100.0

        cl_design = l_val * 3.0 / 20.0   # CL de design
        p = p_code / 20.0                 # position de cambrure max

        # Coefficients standards (serie non reflexe, q=0)
        # Table m, k1 pour differentes valeurs de p
        _table = {
            0.05: (0.0580, 361.400),
            0.10: (0.1260, 51.640),
            0.15: (0.2025, 15.957),
            0.20: (0.2900, 6.643),
            0.25: (0.3910, 3.230),
        }

        if p not in _table:
            raise ValueError(
                u"NACA 5 chiffres : position de cambrure p=%.2f "
                u"non supportee" % p)

        m, k1 = _table[p]

        # Demi-profil
        n_half = n_points // 2
        beta = np.linspace(0, np.pi, n_half)
        x = 0.5 * (1.0 - np.cos(beta))

        # Epaisseur
        yt = 5.0 * t * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )

        # Ligne de cambrure 5 chiffres (serie non reflexe)
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
        front = x <= m
        back = ~front
        yc[front] = (k1 / 6.0) * (
            x[front]**3 - 3.0 * m * x[front]**2
            + m**2 * (3.0 - m) * x[front])
        yc[back] = (k1 * m**3 / 6.0) * (1.0 - x[back])
        dyc[front] = (k1 / 6.0) * (
            3.0 * x[front]**2 - 6.0 * m * x[front]
            + m**2 * (3.0 - m))
        dyc[back] = -(k1 * m**3 / 6.0)

        theta = np.arctan(dyc)

        x_ext = x - yt * np.sin(theta)
        y_ext = yc + yt * np.cos(theta)
        x_int = x + yt * np.sin(theta)
        y_int = yc - yt * np.cos(theta)

        extrados = np.column_stack([x_ext[::-1], y_ext[::-1]])
        intrados = np.column_stack([x_int[1:], y_int[1:]])

        return np.vstack([extrados, intrados])

    # ------------------------------------------------------------------
    #  Ecriture
    # ------------------------------------------------------------------

    def write(self, filepath=None, fmt=None):
        u"""Ecrit le profil dans un fichier.

        :param filepath: chemin de sortie (defaut : self.output_path)
        :type filepath: str or None
        :param fmt: format (defaut : self.output_format)
        :type fmt: str or None
        :returns: chemin du fichier ecrit
        :rtype: str
        """
        if filepath is None:
            filepath = self._output_path
        if filepath is None:
            raise ValueError(
                u"Aucun chemin de sortie specifie "
                u"(filepath ou output_path)")
        if fmt is None:
            fmt = self._output_format

        writers = {
            'selig': self._write_selig,
            'lednicer': self._write_lednicer,
            'csv': self._write_csv,
        }
        if fmt not in writers:
            raise ValueError(
                u"Format d'ecriture inconnu '%s'" % fmt)

        writers[fmt](filepath)
        logger.info(u"Profil '%s' ecrit dans %s (format %s)",
                    self._name, filepath, fmt)
        return filepath

    def _write_selig(self, filepath):
        u"""Ecrit au format Selig."""
        with open(filepath, 'w') as f:
            f.write('%s\n' % self._name)
            for i in range(len(self._points)):
                f.write(' %10.6f %10.6f\n'
                        % (self._points[i, 0], self._points[i, 1]))

    def _write_lednicer(self, filepath):
        u"""Ecrit au format Lednicer."""
        i_le = np.argmin(self._points[:, 0])
        extrados = self._points[:i_le + 1][::-1]  # BA -> BF
        intrados = self._points[i_le:]             # BA -> BF

        with open(filepath, 'w') as f:
            f.write('%s\n' % self._name)
            f.write(' %d.  %d.\n' % (len(extrados), len(intrados)))
            f.write('\n')
            for i in range(len(extrados)):
                f.write(' %10.6f %10.6f\n'
                        % (extrados[i, 0], extrados[i, 1]))
            f.write('\n')
            for i in range(len(intrados)):
                f.write(' %10.6f %10.6f\n'
                        % (intrados[i, 0], intrados[i, 1]))

    def _write_csv(self, filepath):
        u"""Ecrit au format CSV (separateur ;)."""
        with open(filepath, 'w') as f:
            f.write('x;y\n')
            for i in range(len(self._points)):
                f.write('%.6f;%.6f\n'
                        % (self._points[i, 0], self._points[i, 1]))

    # ------------------------------------------------------------------
    #  Trace
    # ------------------------------------------------------------------

    def plot(self, ax=None, show=True):
        u"""Trace le profil.

        :param ax: axes matplotlib existants (None = creation)
        :param show: appeler plt.show() a la fin
        :type show: bool
        :returns: axes matplotlib
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))

        ax.plot(self._points[:, 0], self._points[:, 1],
                'b-', linewidth=1.2)
        ax.set_aspect('equal')
        ax.set_title(self._name)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.grid(True, alpha=0.3)

        # Marquer BA et BF
        le = self.leading_edge
        te = self.trailing_edge
        ax.plot(le[0], le[1], 'ro', markersize=5, label='BA')
        ax.plot(te[0], te[1], 'gs', markersize=5, label='BF')
        ax.legend(fontsize=8)

        if show:
            plt.show()

        return ax


if __name__ == '__main__':
    # Test de base : generer un NACA 2412, normaliser, ecrire en Selig
    # p = Profil.from_naca('2412', n_points=200)
    # print(p)
    # print("Corde = %.2f mm" % p.chord)
    # print("Calage = %.2f deg" % p.calage)
    # print("Epaisseur relative = %.4f" % p.relative_thickness)
    # print("Cambrure relative = %.4f" % p.relative_camber)

    # p.normalize()
    # print("\nApres normalisation :")
    # print(p)
    # print("Corde = %.2f mm" % p.chord)
    # print("Calage = %.2f deg" % p.calage)
    # p.plot()
    # p.rotate(5)  # test rotation
    # print("\nApres rotation :")
    # print(p)
    # print("Corde = %.2f mm" % p.chord)
    # print("Calage = %.2f deg" % p.calage)
    # p.plot()
    # # Ecriture
    # output_file = 'naca2412_normalized.dat'
    # p.write(output_file, fmt='selig')
    p = Profil.from_naca('2412', n_points=200)
    print("points = ", p.points)
    p.normalize()
    print("points apres nomr= ", p.points)
    print("intrados =", p.intrados)
    print("extrados =", p.extrados)
    p.approximate_bezier(degree=3, n_points=50)
    print("bezier extrados points =", p.bezier_extrados.points)
    print("bezier intrados points =", p.bezier_intrados.points)
    