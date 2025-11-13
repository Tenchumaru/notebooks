@ECHO OFF
SETLOCAL

CD /D "%~dp0"
CALL venv\Scripts\activate.bat
IF ERRORLEVEL 1 EXIT /B
jupyter notebook
