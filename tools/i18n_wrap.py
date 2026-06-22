# -*- coding: utf-8 -*-
"""Codemod : enveloppe les arguments-chaines des appels UI Qt avec _().

Opere par tokenisation : on ne reproduit jamais le contenu des chaines
(qui melangent echappements source et litteraux), on insere seulement
``_(`` et ``)`` autour des groupes de chaines qui sont un argument
complet d'un appel dont le nom figure dans la liste blanche.

Usage : python _i18n_wrap.py <fichier.py> [--apply]
Sans --apply : montre le nombre de wraps et n'ecrit rien (dry-run).
"""
import io
import sys
import ast
import tokenize

# Noms d'appels (dernier composant) dont on traduit les arguments-chaines.
WHITELIST = {
    'addMenu', 'addAction', 'setStatusTip', 'setToolTip', 'setText',
    'setWindowTitle', 'showMessage', 'QAction', 'QLabel', 'QPushButton',
    'QCheckBox', 'QGroupBox', 'QRadioButton', 'addRow', 'setTitle',
    'setSuffix', 'setPlaceholderText', 'addItem', 'setPrefix',
    'setLabelText', 'addTab', 'about', 'warning', 'information',
    'critical', 'question',
}

_SKIP = (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
         tokenize.COMMENT, tokenize.ENCODING)


def _wrap_file(path, apply):
    src = open(path, encoding='utf-8').read()
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))

    # Indices des tokens significatifs (on ignore NL/COMMENT/INDENT...).
    sig = [i for i, t in enumerate(toks) if t.type not in _SKIP]
    pos_in_sig = {ti: k for k, ti in enumerate(sig)}

    def name_before_paren(paren_ti):
        """Nom (dernier composant) de l'appel dont '(' est a paren_ti."""
        k = pos_in_sig[paren_ti]
        if k == 0:
            return None
        prev = toks[sig[k - 1]]
        if prev.type == tokenize.NAME:
            return prev.string
        return None

    def enclosing_call_name(group_start_ti):
        """Trouve l'appel englobant en remontant les parentheses."""
        k = pos_in_sig[group_start_ti] - 1  # token sig juste avant le groupe
        depth = 0
        while k >= 0:
            t = toks[sig[k]]
            if t.type == tokenize.OP:
                if t.string in (')', ']', '}'):
                    depth += 1
                elif t.string in ('(', '[', '{'):
                    if depth == 0:
                        if t.string == '(':
                            return name_before_paren(sig[k])
                        return None
                    depth -= 1
            k -= 1
        return None

    wraps = []   # (start_row, start_col, end_row, end_col)
    n = len(sig)
    k = 0
    while k < n:
        ti = sig[k]
        if toks[ti].type == tokenize.STRING:
            # Etendre le groupe de chaines adjacentes (concat implicite).
            start_k = k
            while k + 1 < n and toks[sig[k + 1]].type == tokenize.STRING:
                k += 1
            end_k = k
            start_ti, end_ti = sig[start_k], sig[end_k]

            prev_t = toks[sig[start_k - 1]] if start_k > 0 else None
            next_t = toks[sig[end_k + 1]] if end_k + 1 < n else None

            ok = False
            if prev_t is not None and prev_t.type == tokenize.OP \
                    and prev_t.string in ('(', ','):
                # Argument complet : suivi de ')' ou ','
                if next_t is not None and next_t.type == tokenize.OP \
                        and next_t.string in (')', ','):
                    if prev_t.string == '(':
                        call = name_before_paren(sig[start_k - 1])
                    else:
                        call = enclosing_call_name(start_ti)
                    if call in WHITELIST:
                        # Ignorer les chaines vides.
                        grp_txt = ''.join(toks[sig[j]].string
                                          for j in range(start_k, end_k + 1))
                        try:
                            val = ast.literal_eval(grp_txt)
                        except Exception:
                            val = None
                        if isinstance(val, str) and val != '':
                            ok = True
            if ok:
                sr, sc = toks[start_ti].start
                er, ec = toks[end_ti].end
                wraps.append((sr, sc, er, ec))
        k += 1

    if not wraps:
        print("%s : 0 wrap" % path)
        return 0

    # Points d'insertion : '_(' au debut, ')' a la fin de chaque groupe.
    inserts = []
    for sr, sc, er, ec in wraps:
        inserts.append((sr, sc, '_('))
        inserts.append((er, ec, ')'))
    # Appliquer du bas vers le haut pour ne pas decaler les positions.
    inserts.sort(key=lambda x: (x[0], x[1]), reverse=True)

    lines = src.splitlines(keepends=True)
    for row, col, text in inserts:
        line = lines[row - 1]
        lines[row - 1] = line[:col] + text + line[col:]
    new_src = ''.join(lines)

    # Verifier que le resultat parse.
    ast.parse(new_src)

    if apply:
        with open(path, 'w', encoding='utf-8', newline='') as fh:
            fh.write(new_src)
        print("%s : %d wraps APPLIQUES" % (path, len(wraps)))
    else:
        print("%s : %d wraps (dry-run, ast OK)" % (path, len(wraps)))
    return len(wraps)


if __name__ == '__main__':
    target = sys.argv[1]
    do_apply = '--apply' in sys.argv[2:]
    _wrap_file(target, do_apply)
