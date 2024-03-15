@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: 设置项目根目录和虚拟环境名称
set "PROJECT_DIR=%~dp0"
set "VENV_NAME=venv"

:: 检查虚拟环境是否存在,如果不存在则创建
if not exist "%PROJECT_DIR%%VENV_NAME%" (
    echo 正在创建虚拟环境...
    virtualenv --python=python3.10 "%PROJECT_DIR%%VENV_NAME%"
    call "%PROJECT_DIR%%VENV_NAME%\Scripts\activate.bat"
    echo 正在升级pip...
    python -m pip install --upgrade pip
    echo 正在安装依赖项...
    pip install -r "%PROJECT_DIR%requirements.txt"
    
    :: 创建Output、dataset和reference_audio文件夹
    for %%d in (Output, dataset, reference_audio) do (
        if not exist "%PROJECT_DIR%%%d" mkdir "%PROJECT_DIR%%%d"
    )
    
    deactivate
)

:: 激活虚拟环境
call "%PROJECT_DIR%%VENV_NAME%\Scripts\activate.bat"

:: 询问用户Input子目录的名称
set /p "INPUT_SUBDIR=请输入Input子目录的名称: "

:: 创建Input子目录
if not exist "%PROJECT_DIR%%INPUT_SUBDIR%" mkdir "%PROJECT_DIR%%INPUT_SUBDIR%"

:: 定义执行项目的函数
:execute_project
set "script_name=%~1"
set "input_dir=%~2"
set "reference_dir=%~3"
set "output_dir=%~4"

set /p "project_choice=你想执行 %script_name% 吗? (y/n): "
if /i "!project_choice!"=="y" (
    if "%script_name%"=="recognize.py" (
        set /p "use_reference_audio=你要使用参考音频吗? (y/n): "
        if /i "!use_reference_audio!"=="y" set "input_dir=!reference_dir!"
    )
    if "%script_name%"=="classify.py" (
        set /p "use_reference_audio=你要使用参考音频吗? (y/n): "
        if /i "!use_reference_audio!"=="y" set "input_dir=!reference_dir!"
    )
    python "%PROJECT_DIR%!script_name!" "%PROJECT_DIR%!input_dir!" "%PROJECT_DIR%!output_dir!"
)
exit /b

:: 定义顺序执行项目的函数
:sequential_execution
call :execute_project audio_renamer.py "%INPUT_SUBDIR%" "%INPUT_SUBDIR%"
set /p "next_choice=继续执行下一个项目吗? (y/n): "
if /i "!next_choice!"=="n" goto menu

call :execute_project filter_audio.py "%INPUT_SUBDIR%" reference_audio
set /p "next_choice=继续执行下一个项目吗? (y/n): "
if /i "!next_choice!"=="n" goto menu

call :execute_project recognize.py "%INPUT_SUBDIR%" reference_audio Output
set /p "next_choice=继续执行下一个项目吗? (y/n): "
if /i "!next_choice!"=="n" goto menu

call :execute_project classify.py "%INPUT_SUBDIR%" reference_audio dataset
goto menu

:: 主菜单
:menu
cls
echo 项目菜单
echo --------
echo 1. 顺序执行项目
echo 2. 重命名音频文件
echo 3. 筛选音频(3-10秒)
echo 4. 执行情感推理
echo 5. 音频分类
echo 6. 退出
echo.

set /p "choice=请输入你的选择(1-6): "

if "!choice!"=="1" (
    call :sequential_execution
) else if "!choice!"=="2" (
    call :execute_project audio_renamer.py "%INPUT_SUBDIR%" "%INPUT_SUBDIR%"
) else if "!choice!"=="3" (
    call :execute_project filter_audio.py "%INPUT_SUBDIR%" reference_audio
) else if "!choice!"=="4" (
    call :execute_project recognize.py "%INPUT_SUBDIR%" reference_audio Output
) else if "!choice!"=="5" (
    call :execute_project classify.py "%INPUT_SUBDIR%" reference_audio dataset
) else if "!choice!"=="6" (
    goto exit
) else (
    echo 无效的选择,请重试。
)

echo.
pause
goto menu

:exit
:: 退出虚拟环境
deactivate

echo 项目执行完成。
pause
