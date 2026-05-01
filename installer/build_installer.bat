@echo off
REM ====================================================================
REM  AirfoilTools - Build complet de l'installateur Windows
REM  Lance depuis la racine du projet :  installer\build_installer.bat
REM
REM  Etapes :
REM    1. Recompiler le manuel LaTeX (si pdflatex dispo)
REM    2. Build PyInstaller -> dist\AirfoilTools\
REM    3. Build Inno Setup  -> installer\Output\AirfoilTools-Setup-2.0.exe
REM ====================================================================

setlocal
cd /d "%~dp0\.."
set ROOT=%cd%

echo.
echo === [1/3] Recompilation du manuel utilisateur ===
echo.
where pdflatex >nul 2>&1
if errorlevel 1 (
    echo  pdflatex absent, manuel non recompile (utilise la version existante).
) else (
    pushd docs\manuel
    pdflatex -interaction=nonstopmode -halt-on-error manuel.tex >nul
    pdflatex -interaction=nonstopmode -halt-on-error manuel.tex >nul
    if errorlevel 1 (
        echo  ERREUR : echec compilation du manuel
        popd
        exit /b 1
    )
    del /q *.aux *.log *.toc *.lof *.out 2>nul
    popd
    echo  Manuel recompile : docs\manuel\manuel.pdf
)

echo.
echo === [2/3] Build PyInstaller ===
echo.
if not exist env_py3\Scripts\python.exe (
    echo ERREUR : env_py3\Scripts\python.exe introuvable.
    echo Verifiez que le venv est cree : py -3 -m venv env_py3
    exit /b 1
)
env_py3\Scripts\python.exe -m PyInstaller AirfoilTools.spec --noconfirm
if errorlevel 1 (
    echo ERREUR : echec build PyInstaller
    exit /b 1
)

echo.
echo === [3/3] Build Inno Setup ===
echo.
set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist %ISCC% (
    set ISCC="C:\Program Files\Inno Setup 6\ISCC.exe"
)
if not exist %ISCC% (
    echo ERREUR : Inno Setup 6 introuvable.
    echo Telecharger : https://jrsoftware.org/isdl.php
    exit /b 1
)
%ISCC% installer\AirfoilTools.iss
if errorlevel 1 (
    echo ERREUR : echec build Inno Setup
    exit /b 1
)

echo.
echo ============================================================
echo  Build complet termine avec succes.
echo  Installateur : installer\Output\AirfoilTools-Setup-2.0.exe
echo ============================================================
endlocal
