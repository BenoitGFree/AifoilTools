#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Onglet Profils : affichage et edition interactive des profils."""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QMessageBox,
    QFrame, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal

from model.profil_spline import ProfilSpline
from .profil_canvas import ProfilCanvas


class TabProfils(QWidget):
    """Onglet principal d'edition des profils aerodynamiques."""

    profil_changed = Signal(str)  # role: 'current' ou 'reference'

    def __init__(self, parent=None):
        super().__init__(parent)

        # Profils
        self._profil_current = None
        self._profil_reference = None
        self._profil_flap = None

        self._build_ui()
        self._load_default_profiles()

    # Cadre arrondi pour regrouper les controles (scope par objectName
    # pour ne pas border les QLabel, qui derivent de QFrame).
    _FRAME_STYLE = (
        "QFrame#ctrlGroup { border: 1px solid #b0b0b0;"
        " border-radius: 6px; }")

    def _build_ui(self):
        """Construit l'interface de l'onglet."""
        layout = QVBoxLayout(self)

        # --- Barre de controle : 3 cadres de regroupement ---
        # (1) profil courant  (2) profil reference
        # (3) global : deviation (deux profils) / image (aucun)
        ctrl_layout = QHBoxLayout()

        # === Groupe 1 : Profil courant ===
        grp_current = QFrame()
        grp_current.setObjectName("ctrlGroup")
        grp_current.setStyleSheet(self._FRAME_STYLE)
        h_cur = QHBoxLayout(grp_current)
        h_cur.setContentsMargins(8, 2, 8, 2)

        # Checkbox profil courant
        self._chk_current = QCheckBox("Profil courant")
        self._chk_current.setChecked(True)
        self._chk_current.setToolTip(
            u"Affiche / masque le profil courant (bleu) sur le graphique.")
        self._chk_current.stateChanged.connect(self._on_toggle_current)
        h_cur.addWidget(self._chk_current)

        self._lbl_current = QLabel("NACA 2412")
        self._lbl_current.setStyleSheet("color: #1f77b4; font-weight: bold;")
        self._lbl_current.setToolTip(
            u"Nom du profil courant.\n"
            u"Modifiable via Fichier \u203a Charger profil courant.")
        h_cur.addWidget(self._lbl_current)

        self._chk_porc_current = QCheckBox("Courbure")
        self._chk_porc_current.setChecked(False)
        self._chk_porc_current.setToolTip(
            u"Affiche les porcupines de courbure (segments perpendiculaires"
            u" de longueur proportionnelle a la courbure locale).\n"
            u"Necessite que le profil courant soit en mode Spline.")
        self._chk_porc_current.stateChanged.connect(
            self._on_toggle_porc_current)
        h_cur.addWidget(self._chk_porc_current)

        self._chk_sample_pts = QCheckBox("Pts")
        self._chk_sample_pts.setChecked(False)
        self._chk_sample_pts.setToolTip(
            u"Affiche les points effectivement echantillonnes sur les"
            u" splines (marqueurs x).\n"
            u"Utile pour controler la densite et la repartition.")
        self._chk_sample_pts.stateChanged.connect(
            self._on_toggle_sample_pts)
        h_cur.addWidget(self._chk_sample_pts)

        ctrl_layout.addWidget(grp_current)

        # === Groupe 2 : Profil reference ===
        grp_ref = QFrame()
        grp_ref.setObjectName("ctrlGroup")
        grp_ref.setStyleSheet(self._FRAME_STYLE)
        h_ref = QHBoxLayout(grp_ref)
        h_ref.setContentsMargins(8, 2, 8, 2)

        # Checkbox profil reference
        self._chk_reference = QCheckBox(u"Profil r\u00e9f\u00e9rence")
        self._chk_reference.setChecked(True)
        self._chk_reference.setToolTip(
            u"Affiche / masque le profil de reference (rouge) sur le"
            u" graphique.")
        self._chk_reference.stateChanged.connect(self._on_toggle_reference)
        h_ref.addWidget(self._chk_reference)

        self._lbl_reference = QLabel("NACA 0012")
        self._lbl_reference.setStyleSheet("color: #d62728; font-weight: bold;")
        self._lbl_reference.setToolTip(
            u"Nom du profil de reference.\n"
            u"Modifiable via Fichier \u203a Charger profil reference.")
        h_ref.addWidget(self._lbl_reference)

        self._chk_porc_reference = QCheckBox("Courbure")
        self._chk_porc_reference.setChecked(False)
        self._chk_porc_reference.setToolTip(
            u"Affiche les porcupines de courbure du profil de reference.\n"
            u"Necessite que le profil de reference soit en mode Spline.")
        self._chk_porc_reference.stateChanged.connect(
            self._on_toggle_porc_reference)
        h_ref.addWidget(self._chk_porc_reference)

        ctrl_layout.addWidget(grp_ref)

        # === Groupe 3 : Comparaison & calque ===
        # Deviation : lie aux DEUX profils. Image : liee a AUCUN.
        grp_global = QFrame()
        grp_global.setObjectName("ctrlGroup")
        grp_global.setStyleSheet(self._FRAME_STYLE)
        h_glob = QHBoxLayout(grp_global)
        h_glob.setContentsMargins(8, 2, 8, 2)

        self._chk_deviation = QCheckBox(u"D\u00e9viation")
        self._chk_deviation.setChecked(False)
        self._chk_deviation.setToolTip(
            u"Affiche les porcupines de deviation entre profil courant"
            u" et reference (segments noirs verticaux + enveloppe).\n"
            u"Echelle et densite reglables via le menu contextuel"
            u" (clic droit sur le canvas).")
        self._chk_deviation.stateChanged.connect(
            self._on_toggle_deviation)
        h_glob.addWidget(self._chk_deviation)

        self._chk_image = QCheckBox("Image")
        self._chk_image.setChecked(False)
        self._chk_image.setEnabled(False)
        self._chk_image.setToolTip(
            u"Affiche / masque l'image de calque (en arrière-plan).\n"
            u"Charger une image via Fichier › Charger image de"
            u" calque.\n\n"
            u"Manipulation (maintenir la touche « i ») :\n"
            u"  - i + clic gauche glisser : déplacer l'image\n"
            u"  - i + molette : mettre à l'échelle\n"
            u"  - i + clic droit glisser : tourner autour de (0,0)\n"
            u"  - Maj enfoncée : ajustements fins (ratio 10)")
        self._chk_image.stateChanged.connect(self._on_toggle_image)
        h_glob.addWidget(self._chk_image)

        ctrl_layout.addWidget(grp_global)

        # === Groupe 4 : Flap (volet) ===
        grp_flap = QFrame()
        grp_flap.setObjectName("ctrlGroup")
        grp_flap.setStyleSheet(self._FRAME_STYLE)
        h_flap = QHBoxLayout(grp_flap)
        h_flap.setContentsMargins(8, 2, 8, 2)

        self._chk_flap = QCheckBox("Flap")
        self._chk_flap.setChecked(False)
        self._chk_flap.setToolTip(
            u"Cree et affiche un profil avec volet braque (vert, non"
            u" modifiable), en plus du courant et de la reference.\n"
            u"Le profil braque est construit a partir du profil courant.")
        self._chk_flap.stateChanged.connect(self._on_toggle_flap)
        h_flap.addWidget(self._chk_flap)

        h_flap.addWidget(QLabel("Xf"))
        self._spin_xf = QDoubleSpinBox()
        self._spin_xf.setRange(5.0, 95.0)
        self._spin_xf.setValue(70.0)
        self._spin_xf.setDecimals(1)
        self._spin_xf.setSingleStep(1.0)
        self._spin_xf.setSuffix(" %")
        self._spin_xf.setToolTip(
            u"Position de l'axe d'articulation, en pourcent de corde"
            u" depuis le bord d'attaque.")
        self._spin_xf.valueChanged.connect(self._on_flap_params)
        h_flap.addWidget(self._spin_xf)

        h_flap.addWidget(QLabel("Braquage"))
        self._spin_delta = QDoubleSpinBox()
        self._spin_delta.setRange(-45.0, 45.0)
        self._spin_delta.setValue(0.0)
        self._spin_delta.setDecimals(1)
        self._spin_delta.setSingleStep(1.0)
        self._spin_delta.setSuffix(u" °")
        self._spin_delta.setToolTip(
            u"Angle de braquage du volet en degres.\n"
            u"  positif = bord de fuite vers le haut\n"
            u"  negatif = bord de fuite vers le bas")
        self._spin_delta.valueChanged.connect(self._on_flap_params)
        h_flap.addWidget(self._spin_delta)

        ctrl_layout.addWidget(grp_flap)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # --- Canvas matplotlib ---
        self._canvas = ProfilCanvas(self)
        self._canvas.profil_edited.connect(self._on_current_edited)
        layout.addWidget(self._canvas, stretch=1)

    def _load_default_profiles(self):
        """Charge les profils par defaut au demarrage (mode discret)."""
        self._profil_current = ProfilSpline.from_naca('2412', n_points=150)
        self._profil_current.normalize()

        self._profil_reference = ProfilSpline.from_naca('0012', n_points=150)
        self._profil_reference.normalize()

        self._canvas.set_current_profil(self._profil_current)
        self._canvas.set_reference_profil(self._profil_reference)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_toggle_current(self, state):
        """Affiche/masque le profil courant."""
        self._canvas.set_show_current(state == Qt.Checked.value)

    def _on_toggle_reference(self, state):
        """Affiche/masque le profil de reference."""
        self._canvas.set_show_reference(state == Qt.Checked.value)

    def _on_toggle_porc_current(self, state):
        """Affiche/masque les porcupines du profil courant."""
        checked = state == Qt.Checked.value
        if checked and not self._has_splines(self._profil_current):
            self._warn_curvature_requires_spline()
            self._chk_porc_current.blockSignals(True)
            self._chk_porc_current.setChecked(False)
            self._chk_porc_current.blockSignals(False)
            return
        self._canvas.set_show_porcupines_current(checked)

    def _on_toggle_sample_pts(self, state):
        """Affiche/masque les points echantillonnes."""
        self._canvas.set_show_sample_points(
            state == Qt.Checked.value)

    def _on_toggle_porc_reference(self, state):
        """Affiche/masque les porcupines du profil de reference."""
        checked = state == Qt.Checked.value
        if checked and not self._has_splines(self._profil_reference):
            self._warn_curvature_requires_spline()
            self._chk_porc_reference.blockSignals(True)
            self._chk_porc_reference.setChecked(False)
            self._chk_porc_reference.blockSignals(False)
            return
        self._canvas.set_show_porcupines_reference(checked)

    @staticmethod
    def _has_splines(profil):
        """Retourne True si le profil est defini par des splines."""
        return profil is not None and getattr(profil, 'has_splines', False)

    def _warn_curvature_requires_spline(self):
        """Affiche un avertissement : courbure necessite mode spline."""
        QMessageBox.information(
            self,
            u"Courbure indisponible",
            u"La courbure ne peut etre tracee que sur des profils "
            u"definis par des Beziers (splines).\n\n"
            u"Solutions :\n"
            u"  - Convertir le profil courant via le menu "
            u"« Edition › Convertir en Spline »\n"
            u"  - Charger un profil au format .bspl ou .bez")

    def _on_toggle_deviation(self, state):
        """Affiche/masque la deviation entre profils."""
        self._canvas.set_show_deviation(
            state == Qt.Checked.value)

    def _on_toggle_image(self, state):
        """Affiche/masque l'image de calque."""
        self._canvas.set_background_visible(
            state == Qt.Checked.value)

    # ------------------------------------------------------------------
    # Flap (volet)
    # ------------------------------------------------------------------

    def _on_current_edited(self):
        """Le profil courant a ete edite : propage et recalcule le flap."""
        self.profil_changed.emit('current')
        self._update_flap()

    def _on_toggle_flap(self, state):
        """Active/desactive l'affichage du profil braque."""
        self._update_flap()

    def _on_flap_params(self, _value):
        """Recalcule le profil braque quand Xf ou le braquage change."""
        if self._chk_flap.isChecked():
            self._update_flap()

    def _update_flap(self):
        """Construit (ou retire) le profil braque selon l'etat des controles."""
        if (not self._chk_flap.isChecked()
                or self._profil_current is None):
            self._profil_flap = None
            self._canvas.set_flap_profil(None)
            return
        from model.flap import apply_flap
        try:
            self._profil_flap = apply_flap(
                self._profil_current,
                self._spin_xf.value(),
                self._spin_delta.value())
        except Exception:
            # Echec geometrique : ne pas planter, masquer le profil braque
            self._profil_flap = None
            self._canvas.set_flap_profil(None)
            return
        self._canvas.set_flap_profil(self._profil_flap)
        # Les resultats XFoil du volet (s'il y en a) deviennent obsoletes
        self.profil_changed.emit('flap')

    # ------------------------------------------------------------------
    # Image de calque
    # ------------------------------------------------------------------

    def _set_image_checkbox(self, enabled, checked):
        """Active/coche la checkbox image sans declencher de signal.

        :param enabled: rendre la checkbox active
        :type enabled: bool
        :param checked: etat coche
        :type checked: bool
        """
        self._chk_image.blockSignals(True)
        self._chk_image.setEnabled(enabled)
        self._chk_image.setChecked(bool(checked) if enabled else False)
        self._chk_image.blockSignals(False)

    def load_background_image(self, filepath):
        """Charge une image de calque depuis un fichier.

        :param filepath: chemin du fichier image
        :type filepath: str
        :returns: (True, info) ou (False, message)
        :rtype: tuple
        """
        try:
            w, h = self._canvas.load_background_image(filepath)
        except Exception as e:
            return False, str(e)
        self._set_image_checkbox(True, True)
        return True, "%s (%d x %d)" % (os.path.basename(filepath), w, h)

    def clear_background_image(self):
        """Supprime l'image de calque."""
        self._canvas.clear_background_image()
        self._set_image_checkbox(False, False)

    # ------------------------------------------------------------------
    # Projet (.aftproj)
    # ------------------------------------------------------------------

    def save_project(self):
        """Sauvegarde le projet (profils + image) via un dialogue.

        :returns: (None, None) si annule, (True, chemin), (False, msg)
        :rtype: tuple
        """
        from PySide6.QtWidgets import QFileDialog
        from model.project import (
            save_project, encode_image_array, PROJECT_EXT)

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Enregistrer le projet",
            "projet%s" % PROJECT_EXT,
            u"Projet AirfoilTools (*%s)" % PROJECT_EXT)
        if not filepath:
            return None, None
        if not filepath.lower().endswith(PROJECT_EXT):
            filepath += PROJECT_EXT

        image = None
        if self._canvas.has_background_image:
            state = self._canvas.get_background_state()
            image = {
                'filename': self._canvas.background_filename,
                'source_path': self._canvas.background_source_path,
                'data_b64': encode_image_array(
                    self._canvas.background_array),
            }
            image.update(state)

        try:
            save_project(
                filepath, self._profil_current,
                self._profil_reference, image=image)
        except Exception as e:
            return False, str(e)
        return True, filepath

    def open_project(self, filepath):
        """Ouvre un projet (.aftproj) : profils + image de calque.

        :param filepath: chemin du fichier projet
        :type filepath: str
        :returns: (True, info) ou (False, message)
        :rtype: tuple
        """
        from model.project import load_project, decode_image_b64

        try:
            current, reference, image = load_project(filepath)
        except Exception as e:
            return False, str(e)

        if current is not None:
            self._profil_current = current
            self._lbl_current.setText(current.name)
            self._chk_current.setChecked(True)
            self._canvas.set_current_profil(current)
        if reference is not None:
            self._profil_reference = reference
            self._lbl_reference.setText(reference.name)
            self._chk_reference.setChecked(True)
            self._canvas.set_reference_profil(reference)

        # Image de calque (embarquee dans le projet)
        self._canvas.clear_background_image()
        if image and image.get('data_b64'):
            try:
                arr = decode_image_b64(image['data_b64'])
                self._canvas.set_background_array(
                    arr,
                    filename=image.get('filename'),
                    path=image.get('source_path'))
                self._canvas.set_background_state(image)
                self._set_image_checkbox(
                    True, image.get('visible', True))
            except Exception as e:
                return False, u"Profils OK mais image illisible : %s" % e
        else:
            self._set_image_checkbox(False, False)

        self._update_flap()
        self.profil_changed.emit('current')
        return True, os.path.basename(filepath)

    def load_profil_from_file(self, filepath, role="current"):
        """Charge un profil depuis un fichier.

        :param filepath: chemin du fichier profil
        :type filepath: str
        :param role: 'current' ou 'reference'
        :type role: str
        :returns: (succes, nom_ou_message_erreur)
        :rtype: tuple(bool, str)
        """
        try:
            profil = ProfilSpline.from_file(filepath)
            profil.normalize()
        except Exception as e:
            return False, str(e)

        if role == "reference":
            self._profil_reference = profil
            self._lbl_reference.setText(profil.name)
            self._chk_reference.setChecked(True)
            self._canvas.set_reference_profil(self._profil_reference)
        else:
            self._profil_current = profil
            self._lbl_current.setText(profil.name)
            self._chk_current.setChecked(True)
            self._canvas.set_current_profil(self._profil_current)
            self._update_flap()
        self.profil_changed.emit(role)
        return True, profil.name

    def convert_current_to_spline(self, degree_ext=6, degree_int=6,
                                   max_dev=0.001, max_segments=8,
                                   smoothing=0.0):
        """Convertit le profil courant en mode Spline.

        :param degree_ext: degre pour l'extrados
        :type degree_ext: int
        :param degree_int: degre pour l'intrados
        :type degree_int: int
        :param max_dev: deviation max toleree (pour mode adaptatif)
        :type max_dev: float
        :param max_segments: nombre max de segments par cote
        :type max_segments: int
        :param smoothing: poids de regularisation
        :type smoothing: float
        :returns: (True, nom) si ok, (False, message) si erreur,
                  (None, None) si pas de profil ou deja en Spline
        :rtype: tuple
        """
        p = self._profil_current
        if p is None:
            return None, None
        if p.has_splines:
            return None, None

        try:
            # Extrados
            p.approximate_spline(
                degree=degree_ext, max_dev=max_dev,
                smoothing=smoothing, max_segments=max_segments)
            # Ajuster le degre intrados si different
            if degree_int != degree_ext:
                for seg in p.spline_intrados._segments:
                    current_deg = seg.degree
                    if degree_int > current_deg:
                        seg.elevate(degree_int - current_deg)
                    elif degree_int < current_deg:
                        seg.reduce(current_deg - degree_int)
                p.spline_intrados._invalidate(geometry=True)
        except Exception as e:
            return False, str(e)

        self._canvas.set_current_profil(p)
        self._update_flap()
        self.profil_changed.emit('current')
        return True, p.name

    def _save_profil_dialog(self, profil, title):
        """Sauvegarde un profil via les dialogues partie + format.

        Mutualise le code entre profil courant et profil avec volet.

        :param profil: profil a enregistrer
        :type profil: ProfilSpline
        :param title: titre du dialogue de sauvegarde
        :type title: str
        :returns: (None, None) si annule, (True, info), (False, message)
        :rtype: tuple
        """
        from PySide6.QtWidgets import QFileDialog, QInputDialog

        # 1) Choix de la partie a enregistrer
        parts = [u"Profil complet", u"Extrados seul", u"Intrados seul"]
        choice, ok = QInputDialog.getItem(
            self, "Sauvegarder le profil",
            u"Partie à enregistrer :", parts, 0, False)
        if not ok:
            return None, None
        part = {parts[0]: 'full', parts[1]: 'extrados',
                parts[2]: 'intrados'}[choice]

        # 2) Formats proposes (Lednicer et Spline : profil complet seul)
        if part == 'full':
            flt = (u"Selig (*.dat);;Lednicer (*.dat);;Spline (*.bspl)"
                   u";;CSV (*.csv);;GNU (*.gnu)")
            suffix = ''
        else:
            flt = u"Selig (*.dat);;CSV (*.csv);;GNU (*.gnu)"
            suffix = '_%s' % part

        default = "%s%s.dat" % (profil.name, suffix)
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self, title, default, flt)
        if not filepath:
            return None, None

        if "CSV" in selected_filter:
            fmt = 'csv'
        elif "GNU" in selected_filter:
            fmt = 'gnu'
        elif "Lednicer" in selected_filter:
            fmt = 'lednicer'
        elif "Spline" in selected_filter:
            fmt = 'bspl'
        else:
            fmt = 'selig'

        try:
            profil.write(filepath, fmt=fmt, part=part)
        except Exception as e:
            return False, str(e)

        label = {'full': 'complet', 'extrados': 'extrados',
                 'intrados': 'intrados'}[part]
        return True, "%s (%s)" % (filepath, label)

    def save_current_profil(self):
        """Sauvegarde le profil courant via un dialogue fichier.

        :returns: (None, None) si annule/pas de profil,
                  (True, chemin) si ok, (False, message) si erreur
        :rtype: tuple
        """
        if self._profil_current is None:
            return None, None
        return self._save_profil_dialog(
            self._profil_current, "Sauvegarder le profil courant")

    def save_flap_profil(self):
        """Sauvegarde le profil avec volet (meme procedure que le courant).

        :returns: (None, message) si flap inactif, (None, None) si annule,
                  (True, chemin) si ok, (False, message) si erreur
        :rtype: tuple
        """
        if not self._chk_flap.isChecked() or self._profil_flap is None:
            return None, u"Aucun profil avec volet actif (cocher « Flap »)."
        return self._save_profil_dialog(
            self._profil_flap, "Sauvegarder le profil avec volet")

    @property
    def profil_current(self):
        """Retourne le profil courant (ou None)."""
        return self._profil_current

    @property
    def profil_reference(self):
        """Retourne le profil de reference (ou None)."""
        return self._profil_reference

    @property
    def profil_flap(self):
        """Retourne le profil avec volet braque (ou None si flap inactif)."""
        if not self._chk_flap.isChecked():
            return None
        return self._profil_flap

    def profil_flap_normalized(self):
        u"""Profil avec volet construit dans le repere normalise du courant.

        Le courant est normalise (BA en (0,0), corde 1000, calage 0)
        AVANT d'appliquer le braquage. Le profil resultant est destine a
        XFoil avec ``normalize=False`` : XFoil ne renormalise ni ne
        redresse le braquage, on ne voit donc que l'effet du volet par
        rapport au profil courant.

        :returns: profil braque ou None si flap inactif / echec
        :rtype: ProfilSpline or None
        """
        if not self._chk_flap.isChecked() or self._profil_current is None:
            return None
        from model.flap import apply_flap
        base = type(self._profil_current)(
            self._profil_current.points.copy(),
            name=self._profil_current.name)
        base.normalize()
        try:
            return apply_flap(base, self._spin_xf.value(),
                              self._spin_delta.value())
        except Exception:
            return None

    def change_sampling(self, role='current'):
        """Change le nombre de points d'echantillonnage d'un profil Bezier.

        :param role: 'current' ou 'reference'
        :type role: str
        :returns: (True, info), (False, erreur) ou (None, message)
        :rtype: tuple
        """
        profil = (self._profil_current if role == 'current'
                  else self._profil_reference)
        if profil is None:
            return None, "Pas de profil"
        if not profil.has_splines:
            return None, u"Le profil n'est pas en mode Spline"

        current_n = profil.spline_extrados.n_points
        from PySide6.QtWidgets import QInputDialog
        label = "courant" if role == 'current' else u"r\u00e9f\u00e9rence"
        value, ok = QInputDialog.getInt(
            self,
            u"\u00c9chantillonnage profil %s" % label,
            "Nombre de points :",
            current_n, 10, 10000, 10)
        if not ok:
            return None, None

        profil.spline_extrados.n_points = value
        profil.spline_intrados.n_points = value
        if role == 'current':
            self._canvas.set_current_profil(profil)
            self._update_flap()
        else:
            self._canvas.set_reference_profil(profil)
        self.profil_changed.emit(role)
        return True, "%s : %d points" % (profil.name, value)

    def zoom_fit(self):
        """Zoom adapte (appele depuis le menu)."""
        self._canvas.zoom_fit()
