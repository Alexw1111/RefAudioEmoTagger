@echo off
chcp 65001 >nul
echo Active code page set to UTF-8
echo.

set "ENV_DIR=%~dp0env"
echo Environment directory: "%ENV_DIR%"
echo.

if exist "%ENV_DIR%" (
    echo Environment directory exists.
) else (
    echo Creating environment directory...
    mkdir "%ENV_DIR%"
)
echo.

setlocal enabledelayedexpansion

if not exist "%ENV_DIR%\Scripts\python.exe" (
    echo Installer will automatically download and install Python 3.10.0 to the project's env directory.
    set /p choice="Do you want to continue with Python installation? (y/n): "
    if /i "!choice!" neq "y" exit /b
    
    echo Downloading Python installer...
    curl -o python_installer.exe https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe
    
    echo Installing Python...
    python_installer.exe /quiet PrependPath=1 InstallLauncherAllUsers=1 TargetDir="%ENV_DIR%"
    
    echo Python installation completed.
    echo.
)

endlocal

:: 设置阿里云镜像
echo Setting up Aliyun mirror...
"%ENV_DIR%\Scripts\pip.exe" config set global.index-url https://mirrors.aliyun.com/pypi/simple/
"%ENV_DIR%\Scripts\pip.exe" config set global.trusted-host mirrors.aliyun.com
echo Aliyun mirror configured.
echo.

:: 安装依赖
echo Installing dependencies...
"%ENV_DIR%\Scripts\pip.exe" install -r requirements.txt
echo Dependencies installed.
echo.

echo Script execution completed. Exiting...
