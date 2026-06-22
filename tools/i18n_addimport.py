# -*- coding: utf-8 -*-
"""Insere 'from .i18n import tr as _' apres le dernier import top-level.

Idempotent : ne fait rien si l'import est deja present. Gere les
imports multi-lignes via l'AST (end_lineno). Preserve les fins de
ligne dominantes du fichier.

Usage : python tools/i18n_addimport.py <fichier.py> [<fichier.py> ...]
"""
import ast
import sys

IMPORT_LINE = 'from .i18n import tr as _'


def add_import(path):
    src = open(path, encoding='utf-8').read()
    if IMPORT_LINE in src:
        print("%s : import deja present" % path)
        return
    tree = ast.parse(src)
    last = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last = max(last, node.end_lineno)
    if last == 0:
        print("%s : aucun import top-level trouve, ignore" % path)
        return
    nl = '\r\n' if '\r\n' in src else '\n'
    lines = src.splitlines(keepends=True)
    lines.insert(last, IMPORT_LINE + nl)
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(''.join(lines))
    print("%s : import insere apres la ligne %d" % (path, last))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        add_import(p)
