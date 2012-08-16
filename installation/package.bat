@echo off

rem Specify release version
set version=1.0.1

rem Specify directories
set dirTemp=D:\vISTe\releases
set dirLibs=D:\vISTe\subversion.berlios\installation\libraries
set dirData=D:\vISTe\subversion.berlios\data
set dirTool=D:\vISTe\builds\viste_build_msvc2008

rem Warn user if destination directory not empty
if exist %dirTemp%\%version%\bin echo Installation directory %dirTemp%\%version% not empty!
if exist %dirTemp%\%version%\bin echo If you do not wish to proceed, press Ctrl + C to cancel the installation.
if exist %dirTemp%\%version%\bin pause

rem Specify IDE (msvs2008, qtcreator)
set ide=msvs2008

rem Specify release or debug mode.
set mode=release

rem Check we have a correct output directory
if not exist %dirTool%\tool\bin\%mode% goto errormode

echo Cleaning up folders...
if exist %dirTemp%\%version%\bin\data\72dirs\fibers rmdir /s /q %dirTemp%\%version%\bin\data\72dirs\fibers
if exist %dirTemp%\%version%\bin\data\72dirs\tensors rmdir /s /q %dirTemp%\%version%\bin\data\72dirs\tensors
if exist %dirTemp%\%version%\bin\data\72dirs\sphericalharmonics rmdir /s /q %dirTemp%\%version%\bin\data\72dirs\sphericalharmonics
if exist %dirTemp%\%version%\bin\data\viscontest2010\fibers rmdir /s /q %dirTemp%\%version%\bin\data\viscontest2010\fibers
if exist %dirTemp%\%version%\bin\data\viscontest2010\dti rmdir /s /q %dirTemp%\%version%\bin\data\viscontest2010\dti
if exist %dirTemp%\%version%\bin\data\viscontest2010 rmdir /s /q %dirTemp%\%version%\bin\data\viscontest2010
if exist %dirTemp%\%version%\vcredist_x86.exe del %dirTemp%\%version%\vcredist_x86.exe
if exist %dirTemp%\%version%\bin\data rmdir /s /q %dirTemp%\%version%\bin\data
if exist %dirTemp%\%version%\bin\shaders rmdir /s /q %dirTemp%\%version%\bin\shaders
if exist %dirTemp%\%version%\bin rmdir /s /q %dirTemp%\%version%\bin

echo Creating folders...
mkdir %dirTemp%\%version%\bin
mkdir %dirTemp%\%version%\bin\shaders
mkdir %dirTemp%\%version%\bin\data
mkdir %dirTemp%\%version%\bin\data\viscontest2010
mkdir %dirTemp%\%version%\bin\data\viscontest2010\dti
mkdir %dirTemp%\%version%\bin\data\viscontest2010\fibers
mkdir %dirTemp%\%version%\bin\data\72dirs\fibers
mkdir %dirTemp%\%version%\bin\data\72dirs\tensors
mkdir %dirTemp%\%version%\bin\data\72dirs\sphericalharmonics

echo Copying libraries, settings and microsoft redistribution...
copy %dirLibs%\%mode%\windows\* %dirTemp%\%version%\bin
copy %dirLibs%\..\vcredist_x86.exe %dirTemp%\%version%
copy %dirLibs%\..\settings.xml %dirTemp%\%version%\bin

echo Copying executable and plugins...
if exist %dirTool%\tool\bin\%mode% copy %dirTool%\tool\bin\%mode%\*.exe %dirTemp%\%version%\bin
if exist %dirTool%\tool\bin\%mode% copy %dirTool%\tool\bin\%mode%\*.dll %dirTemp%\%version%\bin

echo Copying shaders...
copy %dirTool%\tool\bin\shaders\*.* %dirTemp%\%version%\bin\shaders

echo Copying sample data...
copy %dirData%\viscontest2010\dti\* %dirTemp%\%version%\bin\data\viscontest2010\dti
copy %dirData%\viscontest2010\fibers\* %dirTemp%\%version%\bin\data\viscontest2010\fibers
copy %dirData%\72dirs\fibers\* %dirTemp%\%version%\bin\data\72dirs\fibers
copy %dirData%\72dirs\tensors\* %dirTemp%\%version%\bin\data\72dirs\tensors
copy %dirData%\72dirs\sphericalharmonics\* %dirTemp%\%version%\bin\data\72dirs\sphericalharmonics
goto end

:errormode
echo Could not find directory (did you choose wrong mode, i.e., release or debug?)
goto end

:end
pause