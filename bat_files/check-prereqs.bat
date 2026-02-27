@echo off
if not exist .\mmc-home.txt ( echo "Do not run this directly" && pause && exit 2 )
if not exist .\neuralnet_models\640m.pt (
	echo.
	echo ***********************************************************************************
	echo ******************************** MISSING FILE *************************************
	echo ###################################################################################
	echo #                                                                                 #
	echo # You are missing the NudeNet neural net weights file.                            #
	echo # This file is called 640m.pt and must be placed                                  #
	echo # in the /neuralnet_models/ folder.                                               #
	echo #                                                                                 #
	echo # Download this file from the NudeNet github at                                   #
	echo # https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.pt    #
	echo #                                                                                 #
	echo # If that link doesn't work, look for the "pytorch link"                          #
	echo # for the "640m" model on the main NudeNet github site:                           #
	echo # https://github.com/notAI-tech/NudeNet                                           #
	echo #                                                                                 #
	echo # Download the 640m.pt file, which should be approximately                        #
	echo # 50MB, and save it in the /neuralnet_models/ folder                              #
	echo #                                                                                 #
	echo # THIS PROGRAM WILL NOT WORK UNTIL YOU DO THIS.                                   #
	echo #                                                                                 #
	echo ###################################################################################
	echo.
	pause
	exit 2
)

REM -- Download ffmpeg if not already present (needed for audio+video recording) --------
if not exist .\tools\ffmpeg.exe (
	echo.
	echo Downloading ffmpeg for audio+video recording support...
	if not exist .\tools mkdir tools
	powershell -Command "Invoke-WebRequest -Uri 'https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-essentials_build.zip' -OutFile ffmpeg-dl.zip"
	powershell -Command "Expand-Archive -Path ffmpeg-dl.zip -DestinationPath ffmpeg-dl-tmp -Force"
	powershell -Command "$src = Get-ChildItem 'ffmpeg-dl-tmp\*\bin\ffmpeg.exe' | Select-Object -First 1; if ($src) { Copy-Item $src.FullName 'tools\ffmpeg.exe' } else { Write-Error 'ffmpeg.exe not found in downloaded archive' }"
	rmdir /s /q ffmpeg-dl-tmp
	del ffmpeg-dl.zip
	echo ffmpeg.exe downloaded to tools\ffmpeg.exe
	echo.
)

