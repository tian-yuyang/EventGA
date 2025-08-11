@echo off
setlocal enabledelayedexpansion

echo Trying to run multiple configuration files (retry on failure)
echo ===================================

set CONFIG_FOLDER=%1

for %%i in (1) do (
    call :process_config %%i
)

echo All done
exit /b

:process_config
set CONFIG_FILE=%1.json
set ATTEMPT=0

echo [%date% %time%] Starting with the config: !CONFIG_FILE!

:retry_file
set /a ATTEMPT+=1
echo Trying !ATTEMPT! times...

python .\run_simulation_gen.py -c "../simulation_args/%CONFIG_FOLDER%/%CONFIG_FILE%"

if !errorlevel! equ 0 (
    echo [Success] !CONFIG_FILE! succeeded at attempt !ATTEMPT!
) else (
    echo [Failure] !CONFIG_FILE! error: !errorlevel! Retrying...
    goto retry_file
)

echo ----------------------------------
exit /b