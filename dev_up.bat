@echo off
rem dev_up.bat - Windows CMD launcher for dev_up.sh

set "BASH_PATH="
if exist "D:\Git\bin\bash.exe" set "BASH_PATH=D:\Git\bin\bash.exe"
if not defined BASH_PATH if exist "C:\Program Files\Git\bin\bash.exe" set "BASH_PATH=C:\Program Files\Git\bin\bash.exe"
if not defined BASH_PATH if exist "C:\Program Files (x86)\Git\bin\bash.exe" set "BASH_PATH=C:\Program Files (x86)\Git\bin\bash.exe"
if not defined BASH_PATH if exist "%LOCALAPPDATA%\Programs\Git\bin\bash.exe" set "BASH_PATH=%LOCALAPPDATA%\Programs\Git\bin\bash.exe"

if not defined BASH_PATH (
    echo [ERROR] Git Bash (bash.exe) was not found. Please install Git for Windows or run dev_up.sh in Git Bash.
    exit /b 1
)

"%BASH_PATH%" "%~dp0dev_up.sh" %*
