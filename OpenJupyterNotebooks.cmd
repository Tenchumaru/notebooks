@ECHO OFF
SETLOCAL

CD /D "%~dp0"
CALL venv\Scripts\activate.bat
jupyter notebook
