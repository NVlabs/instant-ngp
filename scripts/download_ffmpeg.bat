@echo off

set cwd=%cd%
cd /D %~dp0

echo Downloading FFmpeg...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/GyanD/codexffmpeg/releases/download/5.1.2/ffmpeg-5.1.2-full_build.zip', 'ffmpeg.zip')"

echo Unzipping...
powershell Expand-Archive ffmpeg.zip -DestinationPath ..\external\ffmpeg -Force

echo Cleaning up...
if exist ffmpeg.zip del /f /q ffmpeg.zip
exit /b
