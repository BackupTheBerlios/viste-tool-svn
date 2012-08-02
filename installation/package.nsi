; The name of the installer
Name "vIST/e"

; The file to write
OutFile "vISTeInstaller.exe"

; The default installation directory
; InstallDir $PROGRAMFILES\vISTe
InstallDir C:\vISTe

; The registry key of the software
InstallDirRegKey HKLM "Software\vISTe" "Install_Dir"

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
	File /r /x Release /x Debug tool\bin\shaders
	File tool\bin\Release\*.exe
	File tool\bin\Release\*.dll
	File D:\vISTe\subversion.berlios\installation\libraries\release\windows\*.dll
	File D:\vISTe\subversion.berlios\installation\settings.xml
  
	; Write the installation path into the registry
	WriteRegStr HKLM SOFTWARE\vISTe "Install_Dir" "$INSTDIR"
  
	; Write the uninstall keys for Windows
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\vISTe" "DisplayName" "NSIS vIST/e"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\vISTe" "UninstallString" '"$INSTDIR\uninstall.exe"'
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\vISTe" "NoModify" 1
	WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\vISTe" "NoRepair" 1
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
	CreateDirectory "$SMPROGRAMS\vISTe"
	
	; Create a link to the uninstaller
	CreateShortCut "$SMPROGRAMS\vISTe\Uninstall.lnk" "$INSTDIR\uninstall.exe" "" "$INSTDIR\uninstall.exe" 0

	; Change the current directory to "bin" before creating the "vISTe.exe" shortcut.
	; This will set the working directory to the "bin" directory when launching the
	; shortcut, which ensures that the shaders will be loaded correctly.
	
	SetOutPath $INSTDIR\bin
  
	; Create a shortcut for the vIST/e itself
  
	CreateShortCut "$SMPROGRAMS\vISTe\vISTe.lnk" "$INSTDIR\bin\vISTe.exe" "" "$INSTDIR\bin\vISTe.exe" 0
  
SectionEnd


;--------------------------------

; Uninstaller

Section "Uninstall"
  
  ; Remove registry keys
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\vISTe"
  DeleteRegKey HKLM SOFTWARE\vISTe

  ; Remove files and uninstaller
  Delete $INSTDIR\uninstall.exe

  ; Remove shortcuts, if any
  Delete "$SMPROGRAMS\vISTe\*.*"

  ; Remove directories used
  RMDir "$SMPROGRAMS\vISTe"
  RMDir /r "$INSTDIR\bin\shaders"
  RMDir /r "$INSTDIR\bin\data"
  RMDir /r "$INSTDIR\bin"
  RMDir "$INSTDIR"

SectionEnd
