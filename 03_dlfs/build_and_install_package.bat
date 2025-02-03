@echo off

if "%1"=="" (
    echo Usage: install_local_package.bat [path_to_package]
    exit /b 1
)

set PACKAGE_PATH=%1

REM https://packaging.python.org/en/latest/tutorials/packaging-projects/

pip install --upgrade build

python -m build %PACKAGE_PATH%

pip install --upgrade %PACKAGE_PATH%



