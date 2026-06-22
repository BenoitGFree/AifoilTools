# -*- coding: utf-8 -*-
"""Assemble les lots de traduction et genere sources/gui/i18n_en.py.

Verifie la couverture exacte des cles (vs tmp_i18n/untranslated.json)
et la coherence des marqueurs de format (% et \\n) entre cle et valeur.
"""
import glob
import json

untr = json.load(open('tmp_i18n/untranslated.json', encoding='utf-8'))
untr_set = set(untr)

merged = {}
for path in sorted(glob.glob('tmp_i18n/chunk_*.out.json')):
    d = json.load(open(path, encoding='utf-8'))
    merged.update(d)

missing = [k for k in untr if k not in merged]
extra = [k for k in merged if k not in untr_set]

print("Cles attendues : %d" % len(untr_set))
print("Cles traduites : %d" % len(merged))
print("MANQUANTES : %d" % len(missing))
for k in missing[:30]:
    print("   - %r" % k)
print("EN TROP (cle alteree par un agent ?) : %d" % len(extra))
for k in extra[:30]:
    print("   + %r" % k)

# Coherence des marqueurs de format.
nl = chr(10)
warn = 0
for k, v in merged.items():
    if k.count('%') != v.count('%') or k.count(nl) != v.count(nl):
        warn += 1
        if warn <= 30:
            print("FORMAT? cle=%r val=%r" % (k, v))
print("Avertissements format : %d" % warn)

# Generer le module de donnees seulement si couverture complete.
if not missing and not extra:
    body = json.dumps(merged, ensure_ascii=False, indent=4)
    with open('sources/gui/i18n_en.py', 'w', encoding='utf-8',
              newline='') as fh:
        fh.write("# -*- coding: utf-8 -*-\n")
        fh.write('"""Traductions EN Phase 2 (genere par '
                 'tools/i18n_assemble.py).\n\n'
                 'Cle = source francaise exacte. Ne pas editer a la '
                 'main : relancer\nl\'assemblage depuis tmp_i18n/.\n"""\n\n')
        fh.write("EN_PHASE2 = ")
        fh.write(body)
        fh.write("\n")
    print("ECRIT sources/gui/i18n_en.py (%d entrees)" % len(merged))
else:
    print("NON ECRIT : couverture incomplete, corriger d'abord.")
