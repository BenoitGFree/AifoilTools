#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lanceur de l'interface graphique AifoilTools."""

import sys
import os

# Ajouter sources/ au path pour les imports model.* et gui.*
_root = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_root, 'sources')
if _src not in sys.path:
    sys.path.insert(0, _src)

from gui.main_window import main

if __name__ == '__main__':
    main()
