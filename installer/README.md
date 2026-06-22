# Génération de l'installateur Windows

Ce dossier contient la configuration pour produire un installateur
Windows (`AirfoilTools-Setup-<version>.exe`) à partir du build
PyInstaller. La `<version>` provient de `__version__`
(`sources/gui/__init__.py`), reportée dans `AirfoilTools.iss`.

## Prérequis

1. **Inno Setup 6** (gratuit) : https://jrsoftware.org/isdl.php
   - Télécharger `innosetup-6.x.x.exe`
   - Installation par défaut (dans `C:\Program Files (x86)\Inno Setup 6\`)
2. **Build PyInstaller** déjà produit dans `dist/AirfoilTools/`
   (cf. `AirfoilTools.spec` à la racine)

## Procédure complète

### Étape 1 : Build PyInstaller

Depuis la racine du projet :

```bash
env_py3\Scripts\python.exe -m PyInstaller AirfoilTools.spec --noconfirm
```

Vérifier que `dist/AirfoilTools/AirfoilTools.exe` existe et se lance.

### Étape 2 : Build de l'installateur Inno Setup

Deux méthodes :

**Méthode A — Ligne de commande**

```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\AirfoilTools.iss
```

**Méthode B — Interface graphique**

1. Ouvrir Inno Setup Compiler
2. Fichier → Ouvrir → `installer/AirfoilTools.iss`
3. Build → Compile (F9)

### Étape 3 : Tester l'installateur

Le fichier `installer/Output/AirfoilTools-Setup-<version>.exe` (~70 Mo)
peut être :

- Lancé directement pour installer
- Distribué tel quel à un utilisateur final
- Hébergé sur GitHub Releases

## Fonctionnalités de l'installateur produit

- **Multilingue** : Français (par défaut) + Anglais
- **Acceptation de la licence LGPL-3.0**
- **Choix du dossier d'installation** (par défaut : `%LOCALAPPDATA%\Programs\AirfoilTools` en mode utilisateur, `C:\Program Files\AirfoilTools` en mode administrateur)
- **Raccourci menu Démarrer** :
  - AirfoilTools (lance l'application)
  - Manuel utilisateur (ouvre directement le PDF)
  - Désinstallation
- **Raccourci bureau** (optionnel, case à cocher)
- **Désinstallateur automatique** (Panneau de configuration → Programmes)
- **Compression LZMA2 ultra** pour réduire la taille du `.exe`
  d'installation
- **Vérification Windows 10+** avant installation

## Mise à jour de version

La **source unique** de la version est `__version__` dans
`sources/gui/__init__.py` :

```python
__version__ = "3.0"
```

Elle alimente automatiquement :

- la **boîte « À propos »** de la GUI (lecture directe de `__version__`) ;
- le **manuel** : le build PyInstaller (`AirfoilTools.spec`) régénère
  `docs/manuel/version.tex` via `docs/manuel/gen_version.py`, puis
  recompile le PDF.

Seul l'**installateur** doit être mis à jour manuellement (chaîne
d'outils Inno Setup, sans accès à Python) — reporter la même valeur
dans `AirfoilTools.iss` :

```ini
#define MyAppVersion "3.0"
```

**Important** : ne pas changer `AppId` entre deux versions —
Inno Setup l'utilise pour détecter et remplacer les installations
existantes.

## Signature numérique (optionnel)

Pour éviter les avertissements SmartScreen, signer avec un
certificat de signature de code (~200 €/an) :

```ini
[Setup]
SignTool=signtool $f
SignedUninstaller=yes
```

et configurer `signtool` dans Inno Setup :

```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /Ssigntool="signtool sign /f cert.pfx /p PASSWORD /t http://timestamp.digicert.com $f" installer\AirfoilTools.iss
```
