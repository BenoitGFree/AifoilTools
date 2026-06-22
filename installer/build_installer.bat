@echo off
REM ====================================================================
REM  AirfoilTools - Build complet de l'installateur Windows
REM  Lance depuis la racine du projet :  installer\build_installer.bat
REM
REM  Etapes :
REM    1. Build PyInstaller -- dist\AirfoilTools\
REM       (le .spec regenere docs\manuel\version.tex depuis __version__
REM        et recompile les manuels PDF embarques : FR + EN)
REM    2. Build Inno Setup  -- installer\Output\AirfoilTools-Setup-<version>.exe
REM ====================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0\.."

echo.
echo === [1/2] Build PyInstaller ===
echo.
if not exist env_py3\Scripts\python.exe goto :no_venv

env_py3\Scripts\python.exe -m PyInstaller AirfoilTools.spec --noconfirm
if errorlevel 1 goto :pyinstaller_err
goto :step2

:no_venv
echo ERREUR : env_py3\Scripts\python.exe introuvable.
echo Verifiez que le venv est cree :  py -3 -m venv env_py3
exit /b 1

:pyinstaller_err
echo ERREUR : echec du build PyInstaller
exit /b 1


:step2
echo.
echo === [2/2] Build Inno Setup ===
echo.
set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "!ISCC!" set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
if not exist "!ISCC!" goto :no_iscc

"!ISCC!" installer\AirfoilTools.iss
if errorlevel 1 goto :iscc_err

echo.
echo ============================================================
echo  Build complet termine avec succes.
echo  Installateur disponible dans : installer\Output\
echo ============================================================
endlocal
exit /b 0

:no_iscc
echo ERREUR : Inno Setup 6 introuvable.
echo Telecharger : https://jrsoftware.org/isdl.php
exit /b 1

:iscc_err
echo ERREUR : echec du build Inno Setup
exit /b 1
