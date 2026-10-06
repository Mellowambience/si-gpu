$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (!(Test-Path -LiteralPath $vswhere)) { throw 'Visual Studio C++ Build Tools not found' }
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$installation) { throw 'No Visual Studio C++ compiler installation found' }
$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
Push-Location $PSScriptRoot
try {
    @("@echo off", "call `"$vcvars`" >nul", "if errorlevel 1 exit /b 1", "cl /nologo /std:c++17 /EHsc /O2 /W4 baseline.cpp /Fe:baseline.exe /link d3d12.lib dxgi.lib d3dcompiler.lib", "if errorlevel 1 exit /b 1", "cl /nologo /std:c++17 /EHsc /O2 /W4 temporal.cpp /Fe:temporal.exe /link d3d12.lib dxgi.lib d3dcompiler.lib", "exit /b %errorlevel%") | Set-Content -LiteralPath '.build.cmd' -Encoding ascii
    & cmd.exe /d /c .build.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Native baseline compilation failed' }
} finally { Pop-Location }
