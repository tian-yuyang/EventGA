@echo off
setlocal enabledelayedexpansion

echo trying to run multiple configuration files (retry on failure)
echo ===================================

set CONFIG_FOLDER=imk_hard_high

for %%i in (1 2 3) do (
    set CONFIG_FILE=!CONFIG_FOLDER!_%%i.json
    set ATTEMPT=0
    
    echo [%date% %time%] starting with the config: !CONFIG_FILE!
    
    :retry_file
    set /a ATTEMPT+=1
    echo trying !ATTEMPT! times...
    
    python .\run_simulation_high_level.py -c "../simulation_args/!CONFIG_FOLDER!/!CONFIG_FILE!"
    
    if !errorlevel! equ 0 (
        echo [success] !CONFIG_FILE! success at !ATTEMPT! times
    ) else (
        echo [failure] !CONFIG_FILE! error: !errorlevel! retrying...
        timeout /t 5 /nobreak >nul
        goto retry_file
    )
    
    echo ----------------------------------
)

echo all done
endlocal