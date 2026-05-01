; ====================================================================
;  AirfoilTools - Script Inno Setup
;  Genere un installateur Windows standalone a partir du build
;  PyInstaller (dist/AirfoilTools/).
;
;  Compilation :
;    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\AirfoilTools.iss
;  Resultat :
;    installer\Output\AirfoilTools-Setup-2.0.exe
; ====================================================================

#define MyAppName        "AirfoilTools"
#define MyAppVersion     "2.0"
#define MyAppPublisher   "Benoit Gagnaire"
#define MyAppURL         "https://github.com/BenoitGFree/AirfoilTools"
#define MyAppExeName     "AirfoilTools.exe"
#define MyAppDescription "Analyse aerodynamique 2D de profils d'aile"
#define MySourceDir      "..\dist\AirfoilTools"

[Setup]
; --- Identite de l'application ---
; AppId : GUID unique de l'application (NE PAS changer entre versions)
AppId={{8F2A1C4D-7B3E-4F5A-9C6D-1E2B3A4F5C6D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
AppComments={#MyAppDescription}
VersionInfoVersion={#MyAppVersion}.0.0
VersionInfoDescription={#MyAppDescription}
VersionInfoCompany={#MyAppPublisher}

; --- Installation ---
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
DisableProgramGroupPage=auto
OutputDir=Output
OutputBaseFilename={#MyAppName}-Setup-{#MyAppVersion}
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern

; --- Privileges ---
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; --- Architecture ---
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; --- Apparence ---
SetupIconFile=
UninstallDisplayIcon={app}\{#MyAppExeName}
WizardImageStretch=no

; --- Licence ---
LicenseFile=..\LICENSE

; --- Suppression installation precedente ---
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
      GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; --- Bundle PyInstaller complet ---
; Le * recursif copie tout le contenu de dist/AirfoilTools/
; (l'exe + le dossier _internal/ avec dependances, manuel, xfoil)
Source: "{#MySourceDir}\*"; DestDir: "{app}"; \
        Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; --- Menu Demarrer ---
Name: "{group}\{#MyAppName}"; \
      Filename: "{app}\{#MyAppExeName}"; \
      Comment: "{#MyAppDescription}"
Name: "{group}\Manuel utilisateur"; \
      Filename: "{app}\_internal\docs\manuel.pdf"; \
      Comment: "Manuel utilisateur (PDF)"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; \
      Filename: "{uninstallexe}"

; --- Bureau (optionnel) ---
Name: "{autodesktop}\{#MyAppName}"; \
      Filename: "{app}\{#MyAppExeName}"; \
      Tasks: desktopicon

[Run]
; --- Lancer apres installation ---
Filename: "{app}\{#MyAppExeName}"; \
         Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; \
         Flags: nowait postinstall skipifsilent

[UninstallDelete]
; --- Nettoyage des fichiers crees a l'execution ---
; Le cache utilisateur (~/.airfoiltools/) n'est PAS supprime
; pour preserver les profils telecharges.
Type: filesandordirs; Name: "{app}\__pycache__"

[Code]
// Verification pre-installation : Windows 10+
function InitializeSetup(): Boolean;
var
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  if Version.Major < 10 then begin
    MsgBox('AirfoilTools requiert Windows 10 ou superieur.',
           mbError, MB_OK);
    Result := False;
  end else
    Result := True;
end;
