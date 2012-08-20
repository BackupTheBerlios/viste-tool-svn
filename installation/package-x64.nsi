; The name of the installer
Name "vIST/e (64-bits)"

; The file to write
OutFile "vISTeInstaller_x64.exe"

; The default installation directory
; InstallDir $PROGRAMFILES\vISTe
InstallDir C:\vISTe_x64

; The registry key of the software
InstallDirRegKey HKLM "Software\vISTe_x64" "Install_Dir"

LoadLanguageFile "${NSISDIR}\Contrib\Language files\English.nlf"

; Request application privileges for Windows Vista
RequestExecutionLevel user

;--------------------------------

Page components
Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

;--------------------------------

; Install the tool itself, plus all DLLs
Section "vIST/e (required)"

	; Required installation option
	SectionIn RO
  
	; Set output path to the installation directory
	SetOutPath $INSTDIR\bin
  
	; Copy everything in the "bin" directory
	File /r /x Release /x Debug D:\vISTe\builds\viste_build_msvc2010_x64\tool\bin\shaders
	File D:\vISTe\builds\viste_build_msvc2010_x64\tool\bin\Debug\*.exe
	File D:\vISTe\builds\viste_build_msvc2010_x64\tool\bin\Debug\*.dll
	File D:\vISTe\subversion.berlios\installation\libraries\debug\windows\x64\*.dll
	File D:\vISTe\subversion.berlios\installation\settings.xml
  
	; Write the installation path into the registry
	WriteRegStr HKLM SOFTWARE\vISTe_x64 "Install_Dir" "$INSTDIR"
  
	; Write the uninstall keys for Windows
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\vISTe_x64" "DisplayName" "NSIS vIST/e (64-bit)"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\vISTe_x64" "UninstallString" '"$INSTDIR\uninstall.exe"'
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\vISTe_x64" "NoModify" 1
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\vISTe_x64" "NoRepair" 1
	WriteUninstaller "uninstall.exe"
  
SectionEnd

; Install data sets (optional)
Section "Data Sets"

	; Set the output path
	SetOutPath $INSTDIR\bin\data
  
	; This will copy the entiry content of the "Installer\Data" folder to the output  
	File /r D:\vISTe\subversion.berlios\data\*.*
  
SectionEnd


; Start Menu shortcuts (optional)
Section "Start Menu Shortcuts"

	; Create the Start Menu folder
	CreateDirectory "$SMPROGRAMS\vISTe_x64"
	
	; Create a link to the uninstaller
	CreateShortCut "$SMPROGRAMS\vISTe_x64\Uninstall.lnk" "$INSTDIR\uninstall.exe" "" "$INSTDIR\uninstall.exe" 0

	; Change the current directory to "bin" before creating the "vISTe.exe" shortcut.
	; This will set the working directory to the "bin" directory when launching the
	; shortcut, which ensures that the shaders will be loaded correctly.
	
	SetOutPath $INSTDIR\bin
  
	; Create a shortcut for the vIST/e itself
  
	CreateShortCut "$SMPROGRAMS\vISTe_x64\vISTe_x64.lnk" "$INSTDIR\bin\vISTe.exe" "" "$INSTDIR\bin\vISTe.exe" 0
  
SectionEnd


;--------------------------------

; Uninstaller

Section "Uninstall"
  
  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\vISTe_x64"
  DeleteRegKey HKLM SOFTWARE\vISTe_x64

  ; Remove files and uninstaller
  Delete $INSTDIR\uninstall.exe

  ; Remove shortcuts, if any
  Delete "$SMPROGRAMS\vISTe_x64\*.*"

  ; Remove directories used
  RMDir "$SMPROGRAMS\vISTe_x64"
  RMDir /r "$INSTDIR\bin\shaders"
  RMDir /r "$INSTDIR\bin\data"
  RMDir /r "$INSTDIR\bin"
  RMDir "$INSTDIR"

SectionEnd
