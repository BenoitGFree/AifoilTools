#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanceur de l'interface graphique AirfoilTools."""

import sys
import os

if getattr(sys, 'frozen', False):
    # Mode PyInstaller : les modules sont dans le bundle
    _base = sys._MEIPASS
    if _base not in sys.path:
        sys.path.insert(0, _base)
else:
    # Mode developpement
    _root = os.path.dirname(os.path.abspath(__file__))
    _src = os.path.join(_root, 'sources')
    if _src not in sys.path:
        sys.path.insert(0, _src)

from gui.main_window import main

if __name__ == '__main__':
    main()
