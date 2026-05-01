#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture du dialogue UIUC pour le manuel."""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'sources'))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEventLoop, QTimer

from gui.dialog_uiuc import DialogUIUC


OUT_DIR = os.path.join(_ROOT, 'docs', 'manuel', 'images')


def pump(app, ms):
    """Pomp les events pendant ms millisecondes."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = DialogUIUC(role_label=u"courant")
    dlg.show()

    # Attendre que l'index soit charge (max 10s)
    for _ in range(20):
        pump(app, 500)
        if dlg._list.count() > 0:
            break

    # Pre-remplir la recherche pour montrer le filtrage
    dlg._edit_search.setText("naca")
    pump(app, 200)

    # Selectionner un item naca2412 pour montrer la description
    for i in range(dlg._list.count()):
        item = dlg._list.item(i)
        if item.text().startswith("naca2412"):
            dlg._list.setCurrentItem(item)
            break
    pump(app, 200)

    out_path = os.path.join(OUT_DIR, '11_dialog_uiuc.png')
    pix = dlg.grab()
    pix.save(out_path)
    print("OK:", out_path, "size:", pix.width(), "x", pix.height())

    dlg.close()


if __name__ == '__main__':
    main()
