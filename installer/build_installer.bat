@echo off
REM ====================================================================
REM  AirfoilTools - Build complet de l'installateur Windows
REM  Lance depuis la racine du projet :  installer\build_installer.bat
REM
REM  Etapes :
REM    1. Recompiler le manuel LaTeX (si pdflatex dispo)
REM    2. Build PyInstaller -- dist\AirfoilTools\
REM    3. Build Inno Setup  -- installer\Output\AirfoilTools-Setup-2.0.exe
REM ====================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0\.."

echo.
echo === [1/3] Recompilation du manuel utilisateur ===
echo.
where pdflatex >nul 2>&1
if errorlevel 1 goto :no_pdflatex

pushd docs\manuel
pdflatex -interaction=nonstopmode -halt-on-error manuel.tex >nul
if errorlevel 1 goto :pdflatex_err
pdflatex -interaction=nonstopmode -halt-on-error manuel.tex >nul
if errorlevel 1 goto :pdflatex_err
del /q *.aux *.log *.toc *.lof *.out 2>nul
popd
echo  Manuel recompile : docs\manuel\manuel.pdf
goto :step2

:no_pdflatex
echo  pdflatex absent, manuel non recompile.
echo  La version existante de manuel.pdf sera utilisee.
goto :step2

:pdflatex_err
echo  ERREUR : echec de la compilation du manuel
popd
exit /b 1


:step2
echo.
echo === [2/3] Build PyInstaller ===
echo.
if not exist env_py3\Scripts\python.exe goto :no_venv

env_py3\Scripts\python.exe -m PyInstaller AirfoilTools.spec --noconfirm
if errorlevel 1 goto :pyinstaller_err
goto :step3

:no_venv
echo ERREUR : env_py3\Scripts\python.exe introuvable.
echo Verifiez que le venv est cree :  py -3 -m venv env_py3
exit /b 1

:pyinstaller_err
echo ERREUR : echec du build PyInstaller
exit /b 1


:step3
echo.
echo === [3/3] Build Inno Setup ===
echo.
set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "!ISCC!" set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
if not exist "!ISCC!" goto :no_iscc

"!ISCC!" installer\AirfoilTools.iss
if errorlevel 1 goto :iscc_err

echo.
echo ============================================================
echo  Build complet termine avec succes.
echo  Installateur : installer\Output\AirfoilTools-Setup-2.0.exe
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
