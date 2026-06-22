# -*- coding: utf-8 -*-
"""Extrait les cles _() non encore traduites des fichiers GUI.

Parcourt sources/gui/*.py, collecte les arguments-chaines des appels
_(...), retire celles deja presentes dans i18n._TR['en'], et ecrit la
liste (JSON) des cles a traduire dans tmp_i18n/untranslated.json.
"""
import ast
import glob
import json
import os
import sys

sys.path.insert(0, 'sources')
from gui import i18n  # noqa: E402


def keys_in_file(path):
    tree = ast.parse(open(path, encoding='utf-8').read())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == '_' and len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append(arg.value)
    return out


def main():
    already = set(i18n._TR.get('en', {}).keys())
    todo = {}  # cle -> liste de fichiers (pour info)
    for path in sorted(glob.glob('sources/gui/*.py')):
        for k in keys_in_file(path):
            if k and k not in already:
                todo.setdefault(k, []).append(os.path.basename(path))
    keys = sorted(todo.keys())
    os.makedirs('tmp_i18n', exist_ok=True)
    with open('tmp_i18n/untranslated.json', 'w', encoding='utf-8') as fh:
        json.dump(keys, fh, ensure_ascii=False, indent=1)
    print("Cles non traduites : %d" % len(keys))
    print("Ecrit : tmp_i18n/untranslated.json")


if __name__ == '__main__':
    main()
