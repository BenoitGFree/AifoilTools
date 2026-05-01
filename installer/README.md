# Génération de l'installateur Windows

Ce dossier contient la configuration pour produire un installateur
Windows (`AirfoilTools-Setup-2.0.exe`) à partir du build PyInstaller.

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

Le fichier `installer/Output/AirfoilTools-Setup-2.0.exe` (~70 Mo)
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

Modifier dans `AirfoilTools.iss` :

```ini
#define MyAppVersion "2.1"
```

et dans `sources/gui/main_window.py` (boîte "À propos") + le manuel
LaTeX (`\manuelversion`).

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
