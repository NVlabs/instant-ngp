@echo off

set cwd=%cd%
cd /D %~dp0

echo Downloading COLMAP...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/colmap/colmap/releases/download/3.7/COLMAP-3.7-windows-no-cuda.zip', 'colmap.zip')"

echo Unzipping...
powershell Expand-Archive colmap.zip -DestinationPath ..\external\colmap -Force

echo Cleaning up...
if exist colmap.zip del /f /q colmap.zip
exit /b
