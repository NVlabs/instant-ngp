:: Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
::
:: NVIDIA CORPORATION and its licensors retain all intellectual property
:: and proprietary rights in and to this software, related documentation
:: and any modifications thereto.  Any use, reproduction, disclosure or
:: distribution of this software and related documentation without an express
:: license agreement from NVIDIA CORPORATION is strictly prohibited.

@echo off

set cwd=%cd%
cd /D %~dp0

echo Downloading FFmpeg...
powershell -Command "(New-Object Net.WebClient).DownloadFile('https://github.com/GyanD/codexffmpeg/releases/download/5.1.2/ffmpeg-5.1.2-essentials_build.zip', 'ffmpeg.zip')"

echo Unzipping...
powershell Expand-Archive ffmpeg.zip -DestinationPath ..\external\ffmpeg -Force

echo Cleaning up...
if exist ffmpeg.zip del /f /q ffmpeg.zip
exit /b
