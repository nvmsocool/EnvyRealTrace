@echo on
SETLOCAL ENABLEDELAYEDEXPANSION
D:
cd D:\SchoolWork\vk_raytracing_tutorial_KHR\ray_tracing__before\shaders
set converter=D:\VulkanSDK\1.2.189.1\Bin\glslc.exe
for /R %%V in (*.vert *.frag *.comp *.rchit *.rgen *.rmiss) do (
    set in=%%V
    set out=..\spv\%%~nV%%~xV.spv
    echo %%V
    if exist !out! (
        FOR %%i IN (!out!) DO SET outDate=%%~ti
        FOR %%i IN (!in!) DO SET inDate=%%~ti
        
        IF NOT "!inDate!" GEQ "!outDate!" (
            echo   File is up-to-date, skipping...
        ) else (
            echo   File is outdated, compiling...
            del /F /Q !out!
            %converter% %%V -o !out! --target-env=vulkan1.2
            
        )
    ) else (
        echo   File is missing, compiling...
        %converter% %%V -o !out! --target-env=vulkan1.2
    )
    if not exist !out! (
        pause
    )
)