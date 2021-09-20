set converter=D:\VulkanSDK\1.2.189.1\Bin\glslc.exe
for /R %%V in (*.vert *.frag *.comp *.rchit *.rgen *.rmiss) do %converter% %%V -o ..\spv\%%~nV%%~xV.spv --target-env=vulkan1.2
pause